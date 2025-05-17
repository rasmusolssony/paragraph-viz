library(tidyverse)
library(topics)
library(future)
library(ggplot2)
library(plotly)
library(htmltools)
library(pandoc)
library(purrr)

#' This is to get the tokenizer from the python package "transformers"
#' TODO: How should we handle the case where the model is not available?
#' @param modelName (str) The pre-trained model name in the transformers hub.
#' @importFrom reticulate import
#' @return The RObject of the tokenizer.
#' @noRd
get_tokenizer <- function(modelName) {
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
                             ...) {
  embeddings <- text::textEmbed(
    texts = texts,
    model = modelName,
    device = device,
    dim_name = dim_name,
    ...
  )

  embeddings[["tokens"]] <- embeddings[["tokens"]][[1]]

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
decodeToken <- function(aStringList, modelName) {
  tokenizers <- get_tokenizer(modelName)

  # Convert all tokens to ids
  ids <- aStringList %>% tokenizers$convert_tokens_to_ids()
  # Convert all ids to a string of tokens
  output <- ids %>% tokenizers$decode()
  return(output)
}

#### token process ####

#' Check if token is the start of a word.
#' @param token (str) The token to check.
#' @param subword_sign (str) The sign used as the split among subwords.
#' @return A boolean value.
#' @noRd
is_word_start <- function(tokens, i, subword_sign = "##") {
  token <- tokens[i]
  # List of tokens that signal sentence boundaries (or standalone tokens).
  special_tokens <- c("<s>", "</s>", "[CLS]", "[SEP]")

  # Always mark the very first token as a new word.
  if (i == 1) {
    return(TRUE)
  }

  # If the previous token is a special token, mark current token as a new word.
  if (tokens[i - 1] %in% special_tokens) {
    return(TRUE)
  }

  # If the token itself is a special token, mark it as a new word.
  if (token %in% special_tokens) {
    return(TRUE)
  }

  # If the token is a punctuation, mark it as a new word
  if (grepl("^(?:[[:punct:]])+$", token)) {
    return(TRUE)
  }

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

  # Compute word start flags with index awareness.
  word_starts <- sapply(seq_along(tokens), function(i) {
    is_word_start(tokens, i, subword_sign)
  })

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
    summarize(
      start_row = min(index), end_row = max(index),
      .groups = "drop"
    ) %>%
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
combineSubWords <- function(tokens, modelName, normalize = FALSE) {
  subwordIDs <- getSubWordIDs(tokens$tokens, modelName)

  if (is.null(subwordIDs)) {
    return(tokens)
  }

  for (row in seq_len(nrow(subwordIDs))) {
    start <- subwordIDs$start_row[row]
    end <- subwordIDs$end_row[row]
    # Decode the subword tokens to words
    combined_word <-
      decodeToken(tokens$tokens[start:end], modelName)
    # Trim potential whitespaces
    combined_word <- trimws(combined_word)
    # Replace the inital subword with the combined word and set the rest to NA
    tokens$tokens[start:end] <- c(combined_word, rep(NA, end - start))
    # If we want to generalize by tokens in word set n to number of tokens
    # else set n to 1
    if (normalize) {
      n <- end - start + 1
    } else {
      n <- 1
    }
    # Set the words embeddings to the average of the subwords embeddings.
    tokens <- tokens %>%
      mutate(predicted_value =
               replace(predicted_value, start,
                       sum(predicted_value[start:end]) / n))

  }
  # Rename tokens to words
  words <- tokens %>% rename(words = 1)
  # Remove rows with NA tokens (i.e., former subwords)
  words <- tidyr::drop_na(words)
  return(words)
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
    "CLS" = {
      endRow <- rowSEP - 1
    },
    "SEP" = {
      startRow <- rowCLS + 1
    },
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
    which(tokenEmbeddings[[1]] == start_token, arr.ind = TRUE) %>%
    tibble::as_tibble()
  rowSEP <-
    which(tokenEmbeddings[[1]] == end_token, arr.ind = TRUE) %>%
    tibble::as_tibble()

  # Combine the tibbles of row numbers
  rowCLSSEP <- cbind(rowCLS, rowSEP)

  # Set the column names
  names(rowCLSSEP) <- c("CLS", "SEP")

  return(rowCLSSEP)
}

#### Prediction functions ####

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
predictLanguage <- function(embeddings, model, ...) {
  tokenPredictions <- embeddings$tokens %>%
    purrr::map(~ {
      predictions <- .x %>%
        text::textPredict(
          model_info = model,
          texts = NULL,
          word_embeddings = .x %>% select(-1),
          ...
        ) %>%
        rename(predicted_value = 1)

      bind_cols(.x %>% select(tokens), predictions)
    })

  predictions <- embeddings$paragraphs %>%
    text::textPredict(
      model_info = model,
      word_embeddings = .,
      ...
    ) %>%
    rename(predicted_value = 1)

  paragraphPredictions <-
    bind_cols(embeddings$paragraphs %>% select(1), predictions)

  return(
    list(
      "paragraphs" = paragraphPredictions,
      "tokens" = tokenPredictions
    )
  )
}

tokensToWords <- function(tokenContributions, modelName, normalize = FALSE) {
  wordContributions <- tokenContributions %>%
    purrr::map(combineSubWords, modelName, normalize)

  return(wordContributions)
}

wordsToSentences <- function(wordContributions, modelName, normalize = FALSE) {
  # Process each token tibble to create a sentence-level tibble
  sentenceContributions <- wordContributions %>%
    purrr::map(function(words) {
      # Get the rows corresponding to special tokens (CLS/SEP) for this tibble.
      clsSepRows <- getCLSSEPTokenRows(words, modelName)
      # Split the rows so each is processed individually.
      clsSepRows_split <- split(clsSepRows, seq(nrow(clsSepRows)))

      # For each pair of CLS/SEP rows, determine indices, decode tokens,
      # and sum predicted values.
      sentenceTibble <- clsSepRows_split %>%
        purrr::map_dfr(function(row_info) {
          # Get the start and end indices based on your parameters.
          rows <- getStartAndEndRow("both", row_info$CLS, row_info$SEP)
          startRow <- rows[[1]]
          endRow <- rows[[2]]

          # Decode the tokens between the
          # start and end indices to form a sentence.
          sentence <- decodeToken(words$words[startRow:endRow],
                                  modelName) %>% trimws()
          if (normalize) {
            n <- endRow - startRow + 1
          } else {
            n <- 1
          }
          # Sum predicted values for these tokens.
          predicted_value <- sum(words$predicted_value[startRow:endRow]) / n

          tibble(
            sentence = sentence,
            predicted_value = predicted_value,
            start_idx = startRow,
            end_idx = endRow
          )
        })

      # Return the sentence tibble for this token embedding.
      sentenceTibble
    })

  return(sentenceContributions)
}

getContributionScores <- function(
  predictions,
  model,
  modelName,
  referenceValue = NULL,
  globalNormalization = FALSE
) {
  # Get beta0 from the model to calculate contribution score.
  if (is.null(referenceValue)) {
    referenceValue <- model$final_model$fit$fit$fit$a0[[1]]
  }

  tokenContributions <- predictions$tokens %>%
    purrr::map(~ {
      .x <- .x %>%
        mutate(across(predicted_value, ~ . - referenceValue))
      if (!globalNormalization) {
        .x <- .x %>% mutate(across(predicted_value, ~ . / n()))
      }
      return(.x)
    })

  wordContributions <-
    tokensToWords(tokenContributions, modelName, globalNormalization)
  sentenceContributions <-
    wordsToSentences(wordContributions, modelName, globalNormalization)

  contributions <- tibble(
    tokens = tokenContributions,
    words = wordContributions,
    sentences = sentenceContributions,
    paragraphs = predictions$paragraphs,
    referenceValue = referenceValue
  )

  return(contributions)
}

#### Color functions ####

# Define the color gradient
generate_gradient <- function(
  values,
  lower_limit,
  upper_limit,
  palette = NULL
) {
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

pearsonCorrelation <- function(predictions, targets) {
  cor(predictions, targets, method = "pearson")
}

createColoredTibble <- function(predictions, limits, palette = NULL, shapley) {
  # Generate color codes for each value
  predictions$words <- lapply(predictions$words, function(words) {
    if (shapley) {
      upper_limit <- max(abs(unlist(words$predicted_value)))
      lower_limit <- 0 - max(abs(unlist(words$predicted_value)))
    } else {
      upper_limit <- limits[2]
      lower_limit <- limits[1]
    }
    words %>%
      mutate(colorCodes = generate_gradient(
        predicted_value,
        lower_limit, upper_limit,
        palette
      ))
  })
  predictions$sentences <- lapply(predictions$sentences, function(sentences) {
    if (shapley) {
      upper_limit <- max(abs(unlist(sentences$predicted_value)))
      lower_limit <- 0 - max(abs(unlist(sentences$predicted_value)))
    } else {
      upper_limit <- limits[2]
      lower_limit <- limits[1]
    }
    sentences %>%
      mutate(colorCodes = generate_gradient(
        predicted_value,
        lower_limit, upper_limit,
        palette
      ))
  })
  predictions$paragraphs$colorCodes <-
    generate_gradient(
      predictions$paragraphs$predicted_value,
      limits[1], limits[2], palette
    )

  return(predictions)
}

#### Visualization functions ####

library(htmltools)

# Generate HTML for tokens
generate_words_html <- function(words) {
  words <- split(words, seq(nrow(words)))
  token_html <- lapply(words, function(word) {
    span(
      style = paste0(
        "background-color:", word$colorCodes, ";",
        "padding: 0 2px;", # Spacing around each token
        "margin: 0 1px;" # Space between tokens
      ),
      title = paste("Predicted value:", word$predicted_value), # Hover text
      word$words
    )
  })
  return(unname(token_html))
}

generate_words_htmls <- function(tokens) {
  words_htmls <- lapply(tokens, generate_words_html)
  return(words_htmls)
}

# Generate HTML for sentences
generate_sentences_html <- function(sentences, words_html, includeCLSSEP) {
  sentences <- split(sentences, seq(nrow(sentences)))
  sentence_html <- lapply(sentences, function(sentence) {
    if (!includeCLSSEP) {
      start_idx <- sentence$start_idx + 1
      end_idx <- sentence$end_idx - 1
    } else {
      start_idx <- sentence$start_idx
      end_idx <- sentence$end_idx
    }
    div(
      style = paste0(
        "background-color:", sentence$colorCodes, ";",
        "padding: 5px;", # Padding for sentences
        "margin-bottom: 5px;" # Space between sentences
      ),
      title = paste("Predicted value:", sentence$predicted_value), # Hover text
      words_html[start_idx:end_idx]
    )
  })
  do.call(tagList, sentence_html)
}

# Generate paragraph HTML
generate_paragraph_html <- function(paragraph, sentences_html,
                                    target, referenceValue) {
  return(
    div(
      style = paste0(
        "background-color:", paragraph$colorCodes, ";",
        "padding: 10px;"
      ),
      title = paste("Predicted value: ", paragraph$predicted_value),
      sentences_html,
      tags$h3(
        "Predicted score: ",
        trunc(paragraph$predicted_value * 10^2) / 10^2,
        ", True score: ", target,
        ", Reference score: ",
        trunc(referenceValue * 10^2) / 10^2,
      ),
    )
  )
}

generate_legend_html <- function(lower_limit, upper_limit,
                                 palette, title = "Paragraph Legend") {
  legend_values <- seq(lower_limit, upper_limit, length.out = 5)
  legend_colors <- generate_gradient(
    legend_values, lower_limit,
    upper_limit, palette
  )
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
    h3(title),
    legend_html
  )

  return(legend_html)
}

generate_legend_htmls <- function(data, limits, palette) {
  legend_htmls <- lapply(seq(nrow(data$paragraphs)), function(i) {
    upper_limit <- max(abs(unlist(data$sentences[[i]]$predicted_value)))
    upper_limit <- trunc(upper_limit * 10^2) / 10^2

    lower_limit <- 0 - max(abs(unlist(data$sentences[[i]]$predicted_value)))
    lower_limit <- trunc(lower_limit * 10^2) / 10^2

    sentence_legend_html <- generate_legend_html(lower_limit,
      upper_limit,
      palette,
      title = "Sentence Legend"
    )

    upper_limit <- max(abs(unlist(data$words[[i]]$predicted_value)))
    upper_limit <- trunc(upper_limit * 10^2) / 10^2

    lower_limit <- 0 - max(abs(unlist(data$words[[i]]$predicted_value)))
    lower_limit <- trunc(lower_limit * 10^2) / 10^2

    token_legend_html <- generate_legend_html(lower_limit, upper_limit,
      palette,
      title = "Word Legend"
    )

    return(
      div(
        id = paste0("shapley_legend", i),
        style = ifelse(
          i == 1,
          "display: flex; align-items: flex-start; gap: 1rem;",
          "display: none; align-items: flex-start; gap: 1rem;"
        ),
        sentence_legend_html,
        token_legend_html
      )
    )
  })
  return(legend_htmls)
}

generate_histogram_html <- function(data) {
  # Create the histogram using ggplot2
  p <- ggplot2::ggplot(data, aes(x = predicted_value)) +
    geom_histogram(binwidth = 2, fill = "blue", alpha = 0.7, color = "black") +
    labs(
      title = "Histogram of tokens predicted values",
      x = "Predicted Values",
      y = "Frequency"
    ) +
    theme_minimal()

  # Convert ggplot to a plotly object
  histogram_html <- div(
    style = "margin-top: 10px;",
    plotly::ggplotly(p)
  )

  # Save the plot as an HTML widget
  return(histogram_html)
}

generateDocument <- function(
    data,
    targets,
    modelName,
    referenceValue,
    limits,
    palette = NULL,
    shapley = FALSE,
    includeCLSSEP = FALSE,
    filePath = "output.html") {
  # Get the color codes for each value
  data <- createColoredTibble(data, limits, palette, shapley)

  # Generate list of tokens htmls. One for each paragraph.
  words_htmls <-
    generate_words_htmls(data$words)

  # Generate list of sentences htmls. One for each paragraph.
  sentences_htmls <- mapply(
    generate_sentences_html,
    data$sentences, words_htmls, includeCLSSEP, SIMPLIFY = FALSE
  )

  # Generate list of paragraph htmls.
  paragraph_htmls <- mapply(
    function(paragraph, sentences_html, target) {
      generate_paragraph_html(paragraph, sentences_html,
                              target, data$referenceValue[1])
    }, split(data$paragraphs, seq_len(nrow(data$paragraphs))),
    sentences_htmls, split(targets, seq_len(nrow(targets))),
    SIMPLIFY = FALSE
  )

  # Generate the legend(s)
  shapley_legend_htmls <- NULL
  if (shapley) {
    shapley_legend_htmls <- generate_legend_htmls(data, limits, palette)
  }

  legend_html <- generate_legend_html(limits[1], limits[2], palette)

  # Generate list of histogram htmls. One for each paragraph.
  histogram_htmls <- NULL
  if (!shapley) {
    histogram_htmls <- lapply(data$words, generate_histogram_html)
  }

  # Create dropdown menu for paragraph selection
  dropdown_menu <- tags$select(
    id = "paragraphSelector",
    style = "margin-bottom: 10px; line-height: normal; height: 30px;",
    onchange = "showSelectedParagraph()",
    lapply(1:length(paragraph_htmls), function(i) {
      tags$option(value = i, paste("Paragraph", i))
    })
  )
  # Wrap paragraphs and histograms in divs with unique IDs
  paragraph_htmls <- lapply(1:length(paragraph_htmls), function(i) {
    tags$div(
      id = paste0("paragraph", i),
      style = ifelse(i == 1, "display: block;", "display: none;"),
      paragraph_htmls[[i]]
    )
  })
  histogram_htmls <- lapply(1:length(histogram_htmls), function(i) {
    tags$div(
      id = paste0("histogram", i),
      style = ifelse(i == 1, "display: block;", "display: none;"),
      histogram_htmls[[i]]
    )
  })

  # JavaScript to handle paragraph selection
  js_code <- tags$script(HTML("
    function showSelectedParagraph() {
      var selectedParagraph = document
        .getElementById('paragraphSelector').value;
      var paragraphs = document.querySelectorAll('[id^=paragraph]');
      var histograms = document.querySelectorAll('[id^=histogram]');
      var shapley_legends = document.querySelectorAll('[id^=shapley_legend]');
      paragraphs.forEach(function(paragraph) {
        paragraph.style.display = 'none';
      });
      histograms.forEach(function(histogram) {
        histogram.style.display = 'none';
      })
      shapley_legends.forEach(function(legend) {
        legend.style.display = 'none';
      })
      document
        .getElementById('paragraph' + selectedParagraph)
        .style.display = 'block';
      document
        .getElementById('paragraphSelector')
        .style.display = 'block';
      var histogram = document
        .getElementById('histogram' + selectedParagraph)
      if (histogram) {
        histogram.style.display = 'block';
      }
      var legend = document
        .getElementById('shapley_legend' + selectedParagraph)
      if (legend) {
        legend.style.display = 'flex';
      }
    }
  "))

  # Wrap in a basic HTML structure
  full_html <- tags$html(
    tags$head(
      tags$title("Text Visualization"),
      tags$style(HTML("
        /* Adjust margins for headings and paragraphs */
        h1, h2, p {
          margin-top: 5px;
          margin-bottom: 5px;
        }
        h3 {
          margin-top: 0px;
          margin-bottom: 10px;
        }
        /* You can also override body margin if desired */
        body {
          margin: 5px; 
          padding: 5px;
        }
    "))
    ),
    tags$body(
      style = "margin: 5px;",
      tags$div(
        style = "display: flex; justify-content: space-between; align-items: center; gap: 1rem; width: 100%;",
        tags$h1("Text Prediction Visualization"),
        dropdown_menu
      ),
      tags$h2(modelName),
      tags$div(
        style = "margin: 0; padding: 0;",
        paragraph_htmls
      ),
      tags$div(
        style = "display: flex; align-items: flex-start; gap: 0.5rem; margin: 5px; padding: 0;",
        legend_html,
        shapley_legend_htmls
      ),
      histogram_htmls,
      js_code
    )
  )

  # Save the HTML
  save_html(full_html, filePath)

  return(htmltools::browsable(full_html))
}