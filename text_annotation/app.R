library(shiny)
library(bslib)
library(glue)
library(tidyverse)
library(shinyjs)

texts <- readRDS("texts.rds")

texts$words <- lapply(texts$words, function(x) {
  x[, "annotation"] <- 0
  x %>% as_tibble()
})
texts$sentences <- lapply(texts$sentences, function(x) {
  x[, "ranking"] <- 0
  x %>% as_tibble()
})
texts$paragraphs[, "rating"] <- 14

texts$text_index <- 1

ui <- fluidPage(
  titlePanel("Text Annotation Tool"),
  sidebarLayout(
    position = "right",
    sidebarPanel(
      shinyjs::useShinyjs(),
      tags$div(
        style = "display: flex;",
        tags$div(
          style = "margin-right: 10px;",
          actionButton("prev_text", "Previous Text")
        ),
        tags$div(
          style = "margin-right: 10px;",
          actionButton("next_text", "Next Text")
        ),
        downloadButton("save_annotations", "Save Annotations")
      ),
      tags$div(
        uiOutput("rank_info")
      )
    ),
    mainPanel(
      tags$div(
        style = "width: 100%; margin-bottom: 20px; padding: 15px;
                 border: 1px solid #e3e3e3; border-radius: 5px;
                 box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.14)",
        uiOutput("text_display"),
      ),
      tags$div(
        style = "width: 100%; margin-bottom: 20px; padding: 15px;
                 border: 1px solid #e3e3e3; border-radius: 5px;
                 box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.14)",
        uiOutput("instructions")
      )
    )
  ),
  tags$script(HTML(
    "document.addEventListener('DOMContentLoaded', function() {
      Shiny.addCustomMessageHandler('annotationData', function(data) {
        // Save the received annotation data to localStorage.
        localStorage.setItem('annotations', JSON.stringify(data));
        console.log('Annotations saved to localStorage:', data);
      });
      document.body.addEventListener('click', function(event) {
        if (event.target.classList.contains('word')) {
          let annotation = 0;
          if (event.target.style.color === 'red') {
            annotation = -1;
            event.target.style.color = 'green';
          } else if (event.target.style.color === 'green') {
            annotation = 0;
            event.target.style.color = 'black';
          } else {
            annotation = 1;
            event.target.style.color = 'red';
          }
          event.target.setAttribute('data-annotation', annotation);
          } else if (['next_text', 'prev_text', 'save_annotations']
                    .includes(event.target.id)) {
          let words = document.querySelectorAll('.word');
          let data = Array.from(words).map(el => (
              el.getAttribute('data-annotation')
          ));
          Shiny.setInputValue(
            'word_annotations', data, {priority: 'event'}
          );
          // Then, if next/prev, trigger navigation after a short delay.
          if (event.target.id === 'next_text') {
            setTimeout(() => {
              Shiny.setInputValue(
                'next_trigger', Math.random(), {priority: 'event'}
              );
            }, 150); // Adjust delay if needed.
          } else if (event.target.id === 'prev_text') {
            setTimeout(() => {
              Shiny.setInputValue(
                'prev_trigger', Math.random(), {priority: 'event'}
              );
            }, 150);
          }
        }
      });
      $(document).on('shiny:connected', function() {
        const savedAnnotations = localStorage.getItem('annotations');
        if (savedAnnotations) {
          console.log('Restored annotations:', JSON.parse(savedAnnotations));
          Shiny.setInputValue(
           'restored_annotations', 
            JSON.parse(savedAnnotations), 
            {priority: 'event'}
          );
        }
      });
    });"
  ))
)

server <- function(input, output, session) {
  text_index <- reactiveVal(1)
  annotations <- reactiveValues(data = texts)
  observeEvent(text_index(), {
    if (text_index() == 1) {
      shinyjs::disable("prev_text")
    } else {
      shinyjs::enable("prev_text")
    }
    if (text_index() == length(texts$words)) {
      shinyjs::disable("next_text")
    } else {
      shinyjs::enable("next_text")
    }
  })
  output$text_display <- renderUI({
    sentences <- annotations$data$sentences[[text_index()]]
    words <- annotations$data$words[[text_index()]]
    rankings <-
      annotations$data$sentences[[text_index()]]$ranking
    value <-
      annotations$data$paragraphs$rating[[text_index()]]
    word_index <- 0
    sentences_html <- lapply(seq_len(nrow(sentences)), function(i) {
      start_idx <- sentences[i, "start_idx"][[1]]
      end_idx <- sentences[i, "end_idx"][[1]]
      sentence_words <- words$words[start_idx:end_idx]
      word_annotations <- words$annotation[start_idx:end_idx]
      tags$div(
        style = "display: flex; margin: 10px; justify-content: space-between;",
        tags$div(
          style = "flex: left;",
          lapply(seq_along(sentence_words), function(j) {
            word_index <<- word_index + 1
            if (j == 1 || j == length(sentence_words)) {
              word <- ""
            } else{
              word <- sentence_words[j]
            }
            color <-
              if (word_annotations[j] == 1) "red"
              else if (word_annotations[j] == -1) "green"
              else "black"
            tags$span(word, " ", id = word_index,
              class = "word", `data-annotation` = word_annotations[j],
              style = glue("cursor: pointer; font-size: 20px; 
                        user-select: none; color: {color};")
            )
          }),
        ),
        tags$div(
          style = "flex: right; max-width: 120px;",
          selectInput(paste0("sentence_", i), paste("Rate Sentence:"),
            choices = 10:-10,
            selected = rankings[i]
          )
        )
      )
    })
    tags$div(
      tags$h3(
        style = "margin-bottom: 20px;",
        paste0("Paragraph ", text_index(), " / ", length(texts$words))
      ),
      sentences_html,
      tags$div(
        style = "max-width: 120px;",
        numericInput(
          "passage_rating",
          "Rate passage:",
          value = value,
          min = 0,
          max = 27
        )
      )
    )
  })

  output$instructions <- renderUI({
    tags$div(
      tags$h2("Instructions"),
      tags$ul(
        tags$li(
          tags$p(
            tags$strong("Read the passage:"),
            "Start by reading the full paragraph from start to finish before marking or rating."
          )
        ),
        tags$li(
          tags$p(
            tags$strong("Rate the entire passage:"),
            "Try, to the best of your ability, to estimate the rating scale score (PHQ-9 : from 0-27) of the participant, based on the text response you just read."
          )
        ),
        tags$li(
          tags$p(
            tags$strong("Mark individual words:"),
            "If you think a word or sequence of words, accounting for the context in which they are used, is contributing to depression or good mental health (“flourishing”), you can click on each word to mark them. If you click it once it will become red, marking it as depressive, if you click it twice it will become green marking it as healthy and if you click it again it will go back to being neutral. \n
            Please highlight words / sequences accordingly",
            tags$ul(
              tags$li(
                "Red if indicating depression"
              ),
              tags$li(
                "Green if indicating good mental health."
              ),
              tags$li(
                "Unmarked if neutral (indicating neither depression nor good mental health)."
              )
            )
          )
        ),
        tags$li(
          tags$p(
            tags$strong("Rate sentences:"),
            "Next to each sentence is a dropdown menu. Use this menu to rate the sentences in order of how much they contribute to a higher or lower depression score. The scale here is 10 to -10 where 10 means that the sentence indicates that the participant is extremely severely depressed, 0 means it’s neutral or that it doesn’t give any information and -10 means that the sentence indicates that the participant is incredibly flourishing."
          )
        )
      ),
      tags$p(
        tags$strong("Technical details:"),
        " There is a counter on the top of the page that tells you how many paragraphs you have highlighted. Whenever you click 'Next Text', 'Previous Text' or 'Save Annotations' your selections will be saved locally on your browser. This means if you ever close or update the webpage, your progress will be saved for all paragraphs except the one you are currently working on. When you are done, press the 'Save annotations' button, save the file as 'Annotations-<YourName>.rds' and let us know that you are done."
      ),
      tags$p("If you have any questions, don’t hesitate to reach out."),
      tags$p(
        tags$strong("Thank you so much for helping out!")
      )
    )
  })

  output$rank_info <- renderUI({
    tags$div(
      tags$h3(
        "Word Highlighting Legend"
      ),
      tags$p(
        tags$strong(style = "color: Red;", "Red: "),
        "Depressive"
      ),
      tags$p(
        tags$strong(style = "color: Green;", "Green: "),
        "Healthy"
      ),
      tags$h3(
        "Sentence Rating Legend"
      ),
      tags$p(
        tags$strong("10: "),
        "Extremely Severe Depression"
      ),
      tags$p(
        tags$strong("5: "),
        "Moderate Depression"
      ),
      tags$p(
        tags$strong("1: "),
        "Very Mild Depression"
      ),
      tags$p(
        tags$strong("0: "),
        "Neutral"
      ),
      tags$p(
        tags$strong("-1: "),
        "Very Mildly Flourishing"
      ),
      tags$p(
        tags$strong("-5: "),
        "Fairly Flourishing"
      ),
      tags$p(
        tags$strong("-10: "),
        "Incredibly Flourishing"
      )
    )
  })

  output$passage_rating <- renderUI({
    value <-
      annotations$data$paragraphs$rating[[text_index()]]
    numericInput(
      "passage_rating",
      "Rate passage:",
      value = value,
      min = 0,
      max = 27
    )
  })
  output$sentence_ranking <- renderUI({
    rankings <-
      annotations$data$sentences[[text_index()]]$ranking

    lapply(seq_along(rankings), function(i) {
      div(
        style = "flex: 1 1 auto; min-width: 200px;",
        selectInput(paste0("sentence_", i), paste("Rank Sentence", i, ":"),
          choices = seq_along(rankings),
          selected = rankings[i]
        )
      )
    })
  })

  observeEvent(input$restored_annotations, {
    req(input$restored_annotations)
    annotations$data <- input$restored_annotations
    annotations$data$words <- lapply(input$restored_annotations$words,
      function(df) {
        as_tibble(df) %>% mutate(across(everything(), ~ unlist(.)))
      }
    )
    annotations$data$sentences <- lapply(input$restored_annotations$sentences,
      function(df) {
        as_tibble(df) %>% mutate(across(everything(), ~ unlist(.)))
      }
    )

    annotations$data$paragraphs <-
      as_tibble(input$restored_annotations$paragraphs) %>%
      mutate(across(everything(), ~ unlist(.)))
    if (!is.null(annotations$data$text_index)) {
      text_index(annotations$data$text_index[[1]])
    }
  })

  observeEvent(text_index(), {
    annotations$data$text_index <- text_index()
  })

  observeEvent(input$next_trigger, {
    update_annotations()
    text_index(text_index() + 1)
  })

  observeEvent(input$prev_trigger, {
    update_annotations()
    text_index(text_index() - 1)
  })

  observeEvent(input$save_annotations, {
    update_annotations()
  })

  observe({
    session$sendCustomMessage("annotationData", annotations$data)
  })

  update_annotations <- function() {
    req(input$word_annotations)
    annotations$data$words[[text_index()]]$annotation <-
      input$word_annotations

    annotations$data$paragraphs$rating[[text_index()]] <-
      input$passage_rating

    annotations$data$sentences[[text_index()]]$ranking <-
      lapply(seq_along(annotations$data$sentences[[text_index()]]$ranking),
        function(i) {
          input[[paste0("sentence_", i)]]
        }
      )
    annotations$data$text_index <- text_index()
  }

  output$save_annotations <- downloadHandler(
    filename = "annotations.rds",
    content = function(file) {
      update_annotations()
      saveRDS(annotations$data, file)
    }
  )

}

shinyApp(ui, server)
