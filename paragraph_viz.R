# refactor based on the online doc
# Using inline js pack to run via htmltools: https://rstudio.github.io/htmltools/reference/htmlDependency.html

#'Punctuates a string of text.
#' @param str (string) The string to be marked by full stop transformer model.
#' @importFrom reticulate source_python
#' @return A list of sentences
#' @noRd
punctuate_text <- function(str) {
  return(reticulate::fullstopCorrt(str))
}
#' Punctuates a list of strings in a tibble.
#' @param texts (tibble) The tibble contains a list of 
#' texts to be split into sentences.
#' @importFrom furrr future_pmap
#' @return A tibble of the list of sentences
#' @noRd
punctuateTexts <- function(texts) {
  if (!is.null(texts)) {
    text_vector <- as.character(texts[[1]])

    output <- furrr::future_pmap(
      list(text_vector), punctuate_text
    )
  } else {
    print("The input tibble is NULL!")
    output <- NULL
  }

  if (!is.null(output)) {
    return(tibble::as.tibble(t(as.data.frame(output))))
  } else {
    return(NULL)
  }
}

#' This is to get the tokenizer from the python package "transformers"
#' TODO: How should we handle the case where the model is not available?
#' @param modelName (str) The pre-trained model name in the transformers hub.
#' @importFrom reticulate import
#' @return The RObject of the tokenizer.
#' @noRd
getTokenizer <- function(modelName) {
  transformerPack <- reticulate::import("transformers")
  autoTokenizers <- transformerPack$AutoTokenizer
  output <- autoTokenizers$from_pretrained(modelName)
  transformerPack <- NULL
  autoTokenizers <- NULL
  return(output)
}

# TODO: Decide if we want a wrapper function like this or if we want
# to resturcture the code to allow different texts to be used at the
# same time.
#' Create the embeddings for the input texts.
#' @param texts (tibble) The input tibble of texts.
#' @param modelName (str) The pre-trained model name in the transformers hub.
#' @param device (str) The device to use for the embeddings.
#' @param dim_name (logical) Whether to include the dimension names.
#' @param embedSentences (logical) Whether to create embeddings for the
#' sentences. This is only needed if you want to train a model on the sentence
#' level.
#' @param includeCLSSEP (str) To include the embeddings of
#' "CLS", "SEP", "both", or "none" when creating sentence embeddings.
createEmbeddings <- function(texts,
                             modelName = "bert-base-uncased",
                             device = "gpu",
                             dim_name = FALSE,
                             embedSentences = TRUE,
                             includeCLSSEP = "both") {
  embeddings <- text::textEmbed(
    texts = texts,
    model = modelName,
    device = device,
    dim_name = dim_name
  )

  embeddings[["tokens"]] <- embeddings[["tokens"]][[1]]

  if (embedSentences) {
    embeddings <- addSentenceEmbeddings(embeddings, modelName, includeCLSSEP)
  }
  embeddings[["paragraphs"]] <- cbind(texts, embeddings[["texts"]][[1]]) %>%
    tibble::as_tibble()
  return(embeddings)
}

#' Transform the subword tokens back to words,
#' by first transforming the tokens to ids and then back to words.
#' Cannot change the name due to "tokenizers" is a ptr.
#' @param aStringList An input string list.
#' @param tokenizers A tokenizer model from getTokenizer.
#' @param modelName The name of the used model.
#' @importFrom reticulate py_has_attr
#' @return The transformed string.
#' @noRd
decodeToken <- function(aStringList, tokenizers, modelName) {
  # Convert all tokens to ids
  if (reticulate::py_has_attr(tokenizers, "convert_tokens_to_ids")) {
    ids <- aStringList %>% tokenizers$convert_tokens_to_ids()
  } else {
    tokenizers <- getTokenizer(modelName)
    ids <- aStringList %>% tokenizers$convert_tokens_to_ids()
  }
  # Convert all ids to words (combining subwords)
  if (reticulate::py_has_attr(tokenizers, "decode")) {
    output <- ids %>% tokenizers$decode()
  } else {
    tokenizers <- getTokenizer(modelName)
    output <- ids %>% tokenizers$decode()
  }
  return(output)
}

#### token process ####

#' Check if token is the start of a word.
#' @param token (str) The token to check.
#' @param subword_sign (str) The sign used as the split among subwords.
#' @return A boolean value.
#' @noRd
is_word_start <- function(token, subword_sign = "##") {
  if (subword_sign == "##") {
    return(!startsWith(token, subword_sign))
  } else if (subword_sign == "Ġ") {
    return(startsWith(token, subword_sign))
  } else if (subword_sign == "▁") {
    return(!startsWith(token, subword_sign))
  } else {
    stop("The subword sign is not supported!")
  }
}

#' Get the rowIDs of tokens having split sign "##" only for BERT models.
#' @param tokens (tibble) The tokens tibble.
#' @param modelName (str) The pre-trained model name in the transformers hub.
#' The default is "##" for BERT.
#' @importFrom dplyr select
#' @importFrom magrittr %>% %in%
#' @return The rowIDs tibbles.
#' @noRd
getSubWordIDs <- function(tokens, modelName) {
  # Mark the start of each word.
  subword_sign <- get_subword_sign(modelName)
  word_starts <- sapply(tokens, is_word_start,
                        subword_sign)


  # Generate a tibble for processing
  subword_tibble <- tibble::tibble(
    index = seq_along(tokens),
    tokens = tokens,
    is_start = word_starts
  )

  # Identify start and end indices of words
  result <- subword_tibble %>%
    mutate(group = cumsum(is_start)) %>%
    group_by(group) %>%
    summarize(start_row = min(index), end_row = max(index),
              .groups = "drop") %>%
    filter(start_row != end_row) %>%
    select(start_row, end_row)

  return(result)
}

#' Combine the subwords back to words based on the input
#' of function getIDsSubWord.
#' @param tokens (tibble) The tokens tibble.
#' @param subwordIDs (tibble) The output of function getIDsSubWord.
#' @param tokenizer (R_obj) The tokenizer in use.
#' @param modelName (str) The transformer model in use.
#' @importFrom furrr future_pmap
#' @return The tranformed tokens tibble without subword tokens.
#' @noRd
combineSubWords <- function(tokens, subwordIDs, tokenizer, modelName) {
  if (is.null(subwordIDs)) return(tokens)

  for (row in seq_len(nrow(subwordIDs))) {
    start <- subwordIDs$start_row[row]
    end <- subwordIDs$end_row[row]
    # Decode the subword tokens to words
    combined_word <-
      decodeToken(tokens$tokens[start:end], tokenizer, modelName)
    # Trim potential whitespaces
    combined_word <- trimws(combined_word)
    # Replace the inital subword with the combined word and set the rest to NA
    tokens$tokens[start:end] <- c(combined_word, rep(NA, end - start))
    # Set the words embeddings to the average of the subwords embeddings.
    tokens[start, 3:ncol(tokens)] <-
      t(colMeans(tokens[start:end, 3:ncol(tokens)]))
  }
  # Remove rows with NA tokens (i.e., former subwords)
  tokens <- tidyr::drop_na(tokens)
  return(tokens)
}

#' Get the trainable Tb of tokens.
#' @param embeddings (Tibble) Embeddings from text::textEmbed()
#' @param targets (Tibble) The tibble of prediction target
#' @param tokenizer (RObj) The tokenizers to use
#' @param modelName (str) The model name of transformers in use
#' @param combineSubwords (boolean) To combine the subwords or not.
#' The default is TRUE.
#' @importFrom furrr future_pmap_dfr
#' @importFrom dplyr group_by summarise
#' @return The aligned tibble.
#' @noRd
getTrainableWords <- function(embeddings, targets = NULL,
                               modelName, combineSubwords = TRUE) {
  start_token <- get_start_token(modelName)
  end_token <- get_end_token(modelName)

  tokenizer <- getTokenizer(modelName)

  # Combine tokens and targets
  if(!is.null(targets)) {
    tokenTibble <- furrr::future_pmap_dfr(
      list(embeddings, targets[[1]]),
      function(tokens, target) {
        cbind(target = target, tokens)
      }
    )
  } else{
    #TODO: Fix better solution for this
    tokenTibble <- embeddings[[1]]
  }
  # Filter out special tokens (CLS, SEP, etc.)
  tokenTibble <- tokenTibble %>% dplyr::filter(tokens != start_token)
  tokenTibble <- tokenTibble %>% dplyr::filter(tokens != end_token)

  # Combine subwords into words.
  if (combineSubwords) {
    subword_ids <- getSubWordIDs(tokenTibble[["tokens"]], modelName)
    tokenTibble <- combineSubWords(tokenTibble, subword_ids,
                                   tokenizer, modelName)
  }

  return(tokenTibble %>% tibble::as_tibble())
}

#### sentence process ####

#' Gets the start and end row of a sentence based on includeCLSSEP parameter.
#' @param includeCLSSEP The parameter to include the embeddings of
#' "CLS", "SEP", "both", or "none".
#' @param rowCLS The row number of "[CLS]".
#' @param rowSEP The row number of "[SEP]".
#' @return A list of start and end row.
#' @NoRd
getStartAndEndRow <- function(includeCLSSEP, rowCLS, rowSEP) {
  # Determine the start and end rows based on includeCLSSEP parameter
  startRow <- rowCLS
  endRow <- rowSEP

  switch(includeCLSSEP,
    "CLS" = { endRow <- rowSEP - 1 },
    "SEP" = { startRow <- rowCLS + 1 },
    "none" = {
      startRow <- rowCLS + 1
      endRow <- rowSEP - 1
    }
  )

  return(list(startRow, endRow))
}

#' Get the row number of the tokens "[CLS]" and "[SEP]".
#' @param tokenEmbeddings The input tibble of token embeddings.
#' @param modelName The name of the used model to extract exact special tokens.
#' @return A tibble of row number of tokens.
#' @NoRd
getCLSSEPTokenRows <- function(tokenEmbeddings, modelName) {

  # Get the start and end tokens
  start_token <- get_start_token(modelName)
  end_token <- get_end_token(modelName)

  # Check if the input model is supported
  if (is.null(start_token) || is.null(end_token)) {
    stop("Start and/or end token not supported!")
  }

  # Get tibbles with the row numbers of the tokens "[CLS]" and "[SEP]"
  rowCLS <-
    which(tokenEmbeddings[["tokens"]] == start_token, arr.ind = TRUE) %>%
    tibble::as_tibble()
  rowSEP <-
    which(tokenEmbeddings[["tokens"]] == end_token, arr.ind = TRUE) %>%
    tibble::as_tibble()

  # Combine the tibbles of row numbers
  rowCLSSEP <- cbind(rowCLS, rowSEP)

  # Set the column names
  names(rowCLSSEP) <- c("CLS", "SEP")

  return(rowCLSSEP)
}

#' Create a new sentence tibble in line with the token tibble.
#' @param tokenEmbeddings The input tibble of token embeddings.
#' @param modelName The name of the used model to extract exact special tokens.
#' @return A new sentence tibble.
#' @NoRd
initSentenceTibble <- function(tokenEmbeddings, modelName) {
  # Get the rows for CLS and SEP tokens
  clsSepRows <- getCLSSEPTokenRows(tokenEmbeddings, modelName)

  # Initialize a new tibble for sentences
  sentenceTibble <- matrix(nrow = nrow(clsSepRows), 
                           ncol = ncol(tokenEmbeddings)) %>%
    as.data.frame() %>%
    tibble::as_tibble()

  # Initialize the sentence tibble with initial values
  sentenceTibble[seq_len(nrow(clsSepRows)), 1] <- "new"
  sentenceTibble[seq_len(nrow(clsSepRows)), 2:ncol(tokenEmbeddings)] <- 0.0

  # Set column names
  names(sentenceTibble)[1] <- "sentences"
  names(sentenceTibble)[2:ncol(tokenEmbeddings)] <-
    names(tokenEmbeddings)[2:ncol(tokenEmbeddings)]

  # Combine the CLS/SEP rows with the new sentence tibble
  sentenceTibble <- cbind(clsSepRows, sentenceTibble)
  return(sentenceTibble)
}

#' Transform tokens into a sentence.
#' @param sentence The target sentence to transform and the output as well.
#' @param rowCLS The row number of the token "[CLS]".
#' @param rowSEP The row number of the token "[SEP]".
#' @param tokenEmbeddings The input tibble of token embeddings.
#' @param tokenizer The tokenizer used.
#' @param modelName The name of the used tokenizer.
#' @param includeCLSSEP To include the embeddings of
#' "CLS", "SEP", "both", or "none".
#' @return A sentence tibble.
#' @NoRd
transformTokensToSentence <- function(sentence, rowCLS,
                        rowSEP, tokenEmbeddings,
                        tokenizer, modelName, includeCLSSEP) {
  if (!reticulate::py_has_attr(tokenizer, "convert_tokens_to_ids") ||
        !reticulate::py_has_attr(tokenizer, "decode")) {
    tokenizer <- getTokenizer(modelName)
  }
  # Determine the start and end rows based on includeCLSSEP parameter
  rows <- getStartAndEndRow(includeCLSSEP, rowCLS, rowSEP)
  startRow <- rows[[1]]
  endRow <- rows[[2]]

  # Decode the token embeddings to a sentence
  sentence <-
    decodeToken(tokenEmbeddings[["tokens"]][startRow:endRow],
                tokenizer, modelName)

  return(sentence)
}

#' Average the embedding values across tokens.
#' @param sentenceEmbeddings The input embedding value vector.
#' @param rowCLS The row number of "[CLS]".
#' @param rowSEP The row number of "[SEP]".
#' @param tokenEmbeddings The input tibble of token embeddings.
#' @param includeCLSSEP To include the embeddings of
#'  "CLS", "SEP", "both", or "none".
#' @return An integer of the number of sentences.
#' @NoRd
averageTokenEmbeddings <- function(sentenceEmbeddings, rowCLS,
                                   rowSEP, tokenEmbeddings,
                                   includeCLSSEP) {
  # Determine the start and end rows based on includeCLSSEP parameter
  rows <- getStartAndEndRow(includeCLSSEP, rowCLS, rowSEP)
  startRow <- rows[[1]]
  endRow <- rows[[2]]

  # Calculate the average token embeddings in a sentence
  sentenceEmbeddings <-
    tokenEmbeddings[startRow:endRow, 2:ncol(tokenEmbeddings)] %>%
    as.matrix() %>%
    colMeans() %>%
    t() %>%
    tibble::as_tibble()

  # Set the column names
  names(sentenceEmbeddings) <- names(tokenEmbeddings)[2:ncol(tokenEmbeddings)]

  return(sentenceEmbeddings)
}
# TODO: Preserve capital letters.
#' Transforms the tokens to sentences by combining the
#' tokens and averaging the embeddings.
#' @param tokenEmbeddings The input tibble of token embeddings.
#' @param tokenizer The tokenizer used.
#' @param modelName The name of the used tokenizer.
#' @param includeCLSSEP To include the embeddings of
#' "CLS", "SEP", "both", or "none".
#' @importFrom furrr pmap
#' @importFrom furrr pmap_dfr
#' @importFrom future plan
#' @importFrom future cluster
#' @importFrom future future
#' @importForm future value
#' @return A sentence tibble.
#' @NoRd
tokensToSentences <- function(tokenEmbeddings, tokenizer,
                              modelName = "bert-base-uncased",
                              includeCLSSEP = "both") {

  # Ensure the tokenizer has the necessary attributes
  if (!reticulate::py_has_attr(tokenizer, "convert_tokens_to_ids") ||
        !reticulate::py_has_attr(tokenizer, "decode")) {
    tokenizer <- getTokenizer(modelName)
  }

  # Create a new sentence tibble
  sentenceTibble <- initSentenceTibble(tokenEmbeddings, modelName)

  # Generate future sentences
  sentences <- purrr::pmap(
    list(
      sentenceTibble[["sentences"]] %>% as.vector(),
      sentenceTibble[["CLS"]] %>% as.vector(),
      sentenceTibble[["SEP"]] %>% as.vector(),
      tokenEmbeddings %>% list(),
      tokenizer %>% list(),
      modelName %>% list(),
      includeCLSSEP %>% list()
    ),
    transformTokensToSentence
  )

  # Generate future sentence embeddings
  embeddings <- purrr::pmap_dfr(
    list(
      sentenceTibble[, 4:ncol(sentenceTibble)] %>% as.data.frame() %>%
        asplit(1),
      sentenceTibble[["CLS"]] %>% as.vector(),
      sentenceTibble[["SEP"]] %>% as.vector(),
      tokenEmbeddings %>% list(),
      includeCLSSEP %>% list()
    ),
    averageTokenEmbeddings
  )

  # Resolve futures and update the sentence tibble
  sentenceTibble[["sentences"]] <- sentences
  sentenceEmbeddings <- embeddings
  sentenceTibble <- cbind(sentenceTibble[, 1:3], sentenceEmbeddings)
  return(sentenceTibble)
}

#' TODO: DOES THIS ACTUALLY REMOVE SPECIAL TOKEN COLUMNS?
#' Remove columns of special tokens.
#' @param sentenceEmbeddings The output from token2Sent_getSent.
#' @return Text embeddings without special token columns.
#' @NoRd
removeSpecialTokenColumns <- function(sentenceEmbeddings) {
  return(sentenceEmbeddings[, 3:ncol(sentenceEmbeddings)])
}

#' Get the tibble of sentence embeddings.
#' @param tokenList The input list of token embeddings.
#' @param tokenizer The tokenizer used in function textEmbed 
#' to get token embeddings.
#' @param modelName The name of the used tokenizer.
#' @param includeCLSSEP To include the embeddings of "start", 
#' "end", "both", or "none".
#' @return A list containing token embeddings of the function textEmbed()
#'  along with sentence embeddings.
#' @NoRd
addSentenceEmbeddings <- function(embeddings,
                                  modelName = "bert-base-uncased",
                                  includeCLSSEP = "both") {

  tokenizer <- getTokenizer(modelName)

  # Generate sentence embeddings
  sentenceEmbeddings <- purrr::pmap(
    list(
      embeddings[["tokens"]] %>% as.vector(),
      tokenizer %>% list(),
      modelName,
      includeCLSSEP
    ),
    tokensToSentences
  )

  # Remove CLS and SEP columns
  sentenceEmbeddings <- purrr::pmap(
    list(
      sentenceEmbeddings %>% as.vector()
    ),
    removeSpecialTokenColumns
  )

  # Append the combined sentence embeddings to the original list
  embeddings <- append(
    embeddings, 
    list("sentences" = lapply(sentenceEmbeddings, tibble::as_tibble)))

  return(embeddings)
}

#' Get the trainable tibble of sentences.
#' @param embeddings The output from addSentenceEmbeddings.
#' @param targets The targets to predict.
#' @importFrom furrr future_pmap_dfr
#' @return The tibble of sentence embeddings.
#' @NoRd
getTrainableSentences <- function(embeddings, targets) {
  if(!is.null(targets)) {
    sentenceTibble <- furrr::future_pmap_dfr(
      list(embeddings, targets[[1]]),
      function(sentences, target) {
        cbind(target = target, sentences)
      }
    )
  }
  return(sentenceTibble %>% tibble::as_tibble())
}

#### Paragraph processing ####

#' Get the trainable tibble of paragraphs.
#' @param embeddings (R obj) The output of textEmbed()
#' @param targets (vector) The targets to predict
#' @import furrr furrr_pmap
#' @return The trained model of passages and their prediction targets.
#' @NoRd
getTrainableParagraphs <- function(embeddings, targets) {
  if(!is.null(targets)) {
    paragraphTibble <- cbind(targets, embeddings) %>%
      tibble::as_tibble()
  }

  return(paragraphTibble)
}

#### Training functions

#' Train the language model for tokens, sentences, or paragraphs.
#' @param tibble The tibble of embeddings and targets.
#' @retrun The trained model
#' @NoRd
train <- function(tibble, mixture = c(0)) {
  # Train the model
  model <- text::textTrainRegression(
    x = tibble[, 3:ncol(tibble)],
    y = tibble[, 1],
    impute_missing = TRUE,
    #mixture = mixture,
  )

  return(model)
}

#' Train the language model.
#' @param embeddings (R_obj) An R obj containing the information of
#' "token", "sentence",  and "paragraph".
#' @param targets (R_obj) The training target.
#' @param languageLevel (str) "token", "sentence", "paragraph" or "all".
#'  The default is "all".
#' @param modelName (str) The transformer model in use.
#' @importFrom future future value
#' @return The trained model
#' @NoRd
trainLanguageModel <- function(embeddings, targets, languageLevel = "paragraph",
                               modelName, mixture = 0) {
  if (!(languageLevel %in% c("sentence", "token", "paragraph", "all"))) {
    languageLevel <- "paragraph"
  }

  switch(languageLevel,
    "token" = {
      futureTokenModel <- future::future(
        getTrainableWords(embeddings[["tokens"]], targets, modelName) %>%
          train(mixture)
      )
      tokenModel <- future::value(futureTokenModel)
      return(list("tokenModel" = tokenModel))
    },
    "sentence" = {
      futureSentenceModel <- future::future(
        getTrainableSentences(embeddings[["sentences"]], targets) %>%
          train(mixture)
      )
      sentenceModel <- future::value(futureSentenceModel)
      return(list("sentenceModel" = sentenceModel))
    },
    "paragraph" = {
      futureParagraphModel <-
        getTrainableParagraphs(embeddings[["paragraphs"]], targets) %>%
        train(mixture)

      paragraphModel <- futureParagraphModel
      return(list("paragraphModel" = paragraphModel))
    },
    "all" = {

      futureTokenModel <- future::future(
        getTrainableWords(embeddings[["tokens"]], targets, modelName) %>%
          train(mixture)
      )
      futureSentenceModel <- future::future(
        getTrainableSentences(embeddings[["sentences"]], targets) %>%
          train(mixture)
      )
      futureParagraphModel <- future::future(
        getTrainableParagraphs(embeddings[["paragraphs"]], targets) %>%
          train(mixture)
      )
      tokenModel <- future::value(futureTokenModel)
      sentenceModel <- future::value(futureSentenceModel)
      paragraphModel <- future::value(futureParagraphModel)

      return(list(
        "tokenModel" = tokenModel,
        "sentenceModel" = sentenceModel,
        "paragraphModel" = paragraphModel
      ))
    }
  )
}

#### Prediction functions ####
#' Get the predictions of tokens, sentences, or paragraphs.
#' @param embeddings An R obj containing the embeddings of
#' "token", "sentence",  or "passage".
#' @param model A trained token model.
#' @param modelName (str) The transformer model.
#' @return The token predictions
#' @NoRd
predict <- function(embeddings, model, modelName) {
  # Predict the values
  print(model)
  predictions <- text::textPredict(
    model,
    embeddings[, 2:ncol(embeddings)]
  )

  # Combine the predictions with the embeddings
  predictions <- predictions %>% tibble::as_tibble()
  colnames(predictions)[1] <- c("predicted_value")
  predictions <- cbind(embeddings[, 1], predictions)

  return(predictions %>% tibble::as_tibble())
}

#' Get the predictions of the language model.
#' @param embeddings An R obj containing the embeddings of
#' "token", "sentence",  and "paragraph".
#' @param models The models from trainLangaueModel.
#' @param languageLevel "token", "sentence", "parapgraph", "all".
#' The default is "sentence".
#' @param modelName (str) The name of the transformer model.
#' @importFrom future future
#' @return The prediction R object
#' @NoRd
predictLanguage <- function(embeddings, models, languageLevel = "all",
                            modelName) {
  if (!(languageLevel %in% c("sentence", "token", "paragraph", "all"))) {
    languageLevel <- "all"
  }

  switch(languageLevel,
    "token" = {
      tokenPredictions <-
        getTrainableWords(embeddings[["tokens"]], NULL, modelName) %>%
        predict(models[["paragraphModel"]])
      return(list("tokens" = tokenPredictions))
    },
    "sentence" = {
      sentencePredicitons <-
        predict(embeddings[["sentences"]][[1]], models[["paragraphModel"]])
      return(list("sentences" = sentencePredicitons))
    },
    "paragraph" = {
      paragraphPredictions <-
        predict(embeddings[["paragraphs"]], models[["paragraphModel"]])
      return(list("paragraphs" = paragraphPredictions))
    },
    "all" = {
      tokenPredictions <-
        getTrainableWords(embeddings[["tokens"]], NULL, modelName) %>%
        predict(models[["paragraphModel"]])
      sentencePredicitons <-
        predict(embeddings[["sentences"]][[1]], models[["paragraphModel"]])
      paragraphPredictions <-
        predict(embeddings[["paragraphs"]], models[["paragraphModel"]])
      return(list(
        "tokens" = tokenPredictions,
        "sentences" = sentencePredicitons,
        "paragraphs" = paragraphPredictions
      ))
    }
  )
}

#### Color functions ####

# Define the color gradient
generate_gradient <- function(values, lower_limit, upper_limit, palette = NULL) {
  # Default to a red-yellow-blue color palette
  if (is.null(palette)) {
    palette <- "Temps"
  }
  # Ensure limits are numeric
  if (!is.numeric(values) ||
        !is.numeric(lower_limit) ||
        !is.numeric(upper_limit)) {
    stop("Values and limits must be numeric.")
  }

  # Clamp values to the defined limits
  values_clamped <- pmax(lower_limit, pmin(values, upper_limit))

  # Normalize values to a 0-1 scale
  normalized_values <-
    (values_clamped - lower_limit) / (upper_limit - lower_limit)

  # Create the color palette from green to red
  # palette <- colorRampPalette(c("#2ad587", "#0078d2"))
  palette <- colorspace::divergingx_hcl(100, palette)

  # Generate colors for the normalized values
  colors <- palette[ceiling(normalized_values * 99) + 1]

  return(colors)
}

createColoredTibble <- function(predictions, limits, palette = NULL) {
  # Generate color codes for each value
  predictions$tokens$colorCodes <-
    generate_gradient(predictions$tokens$predicted_value,
                      limits[1], limits[2], palette)

  predictions$sentences$colorCodes <-
    generate_gradient(predictions$sentences$predicted_value,
                      limits[1], limits[2], palette)

  predictions$paragraphs$colorCodes <-
    generate_gradient(predictions$paragraphs$predicted_value,
                      limits[1], limits[2], palette)

  return(predictions)
}

#### Visualization functions ####

library(htmltools)

# Generate HTML for tokens
generate_tokens_html <- function(tokens) {
  tokens <- split(tokens, seq(nrow(tokens)))
  token_html <- lapply(tokens, function(token) {
    span(
      style = paste0(
        "background-color:", token$colorCodes, ";",
        "padding: 0 2px;", # Spacing around each token
        "margin: 0 1px;"   # Space between tokens
      ),
      title = paste("Predicted value:", token$predicted_value), # Hover text
      token$tokens
    )
  })
  do.call(tagList, token_html)
}

# Generate HTML for sentences
generate_sentences_html <- function(sentences, tokens_html) {
  sentences <- split(sentences, seq(nrow(sentences)))
  sentence_html <- mapply(function(sentence, content) {
    div(
      style = paste0(
        "background-color:", sentence$colorCodes, ";",
        "padding: 5px;", # Padding for sentences
        "margin-bottom: 5px;" # Space between sentences
      ),
      title = paste("Predicted value:", sentence$predicted_value), # Hover text
      content
    )
  }, sentences, tokens_html, SIMPLIFY = FALSE)
  do.call(tagList, sentence_html)
}

# Generate paragraph HTML
generate_paragraph_html <- function(paragraph, sentences_html) {
  div(
    style = paste0(
      "background-color:", paragraph$colorCodes, ";",
      "padding: 10px;", # Padding for the paragraph
      "margin-bottom: 10px;" # Space between paragraphs
    ),
    title = paste("Predicted value: ", paragraph$predicted_value), # Hover text
    sentences_html
  )
}

generate_legend_html <- function(limits, palette) {
  legend_values <- seq(limits[1], limits[2], length.out = 5)
  legend_colors <- generate_gradient(legend_values, limits[1],
                                     limits[2], palette)
  legend_html <- lapply(1:5, function(i) {
    span(
      style = paste0(
        "background-color:", legend_colors[i], ";",
        "padding: 5px;", # Padding for the legend
        "margin-right: 5px;" # Space between legend items
      ),
      legend_values[i]
    )
  })

  legend_html <- div(
    style = "margin-top: 10px;", # Space above the legend
    h3("Legend"),
    legend_html
  )

  return(legend_html)
}

generate_histogram_html <- function(targets) {
  # Create the histogram using ggplot2
  p <- ggplot2::ggplot(targets, aes(x = values)) +
    geom_histogram(binwidth = 2, fill = "blue", alpha = 0.7, color = "black") +
    labs(
      title = "Histogram of Values",
      x = "Values",
      y = "Frequency"
    ) +
    theme_minimal()

  # Convert ggplot to a plotly object
  p_interactive <- plotly::ggplotly(p)

  # Save the plot as an HTML widget
  return (htmltools::tags$div(
    HTML(htmlwidgets::saveWidget(p_interactive, "temp_plot.html", selfcontained = TRUE)),
    style = "width: 100%; height: 500px;"
  ))
}

generateDocument <- function(
  data,
  embeddings,
  target,
  modelName,
  limits,
  palette = NULL,
  filePath = "output.html"
) {
  #Get the color codes for each value
  data <- createColoredTibble(data, limits, palette)
  # Get the rows for the CLS and SEP tokens
  rows <- getCLSSEPTokenRows(embeddings[["tokens"]][[1]], modelName)
  rows <- split(rows, seq(nrow(rows)))

  # Generate the full document
  subword_sign <- get_subword_sign(modelName)
  word_starts <- sapply(embeddings[["tokens"]][[1]][["tokens"]], is_word_start,
                        subword_sign)
  start <- 1

  tokens_html <- lapply(rows, function(row) {
    # Get the CLS and SEP range for this sentence
    range <- getStartAndEndRow(includeCLSSEP = "none", row$CLS, row$SEP)

    # Calculate the end index for this range
    end <- start + sum(word_starts[range[[1]]:range[[2]]]) - 1

    # Subset tokens for this range
    tokens <- data$tokens[start:end, ]

    # Update start for the next iteration
    start <<- end + 1

    # Generate HTML for these tokens
    return(generate_tokens_html(tokens))
  })
  sentences_html <- generate_sentences_html(data$sentences, tokens_html)

  paragraph_html <- generate_paragraph_html(data$paragraph, sentences_html)

  legend_html <- generate_legend_html(limits, palette)

  # Wrap in a basic HTML structure
  full_html <- tags$html(
    tags$head(tags$title("Text Visualization")),
    tags$body(
      tags$h1("Text Prediction Visualization"),
      paragraph_html,
      legend_html,
      tags$h3("Predicted score: ",
              trunc(data$paragraph$predicted_value * 10^2) / 10^2,
              ", True score: ", target),
    )
  )

  # Save the HTML
  save_html(full_html, filePath)

  return(htmltools::browsable(full_html))

}