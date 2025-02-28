library(shiny)
library(bslib)
library(glue)
library(tidyverse)
library(shinyjs)

texts <- readRDS("texts.rds")

texts$tokens <- lapply(texts$tokens, function(x) {
  x[, "annotation"] <- 0
  x %>% as_tibble()
})
texts$sentences <- lapply(texts$sentences, function(x) {
  x[, "ranking"] <- 0
  x
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
  # observeEvent(text_index(), {
  #   if (text_index() == 1) {
  #     shinyjs::disable("prev_text")
  #   } else {
  #     shinyjs::enable("prev_text")
  #   }
  #   if (text_index() == length(texts$tokens)) {
  #     shinyjs::disable("next_text")
  #   } else {
  #     shinyjs::enable("next_text")
  #   }
  # })
  output$text_display <- renderUI({
    words_by_sentence <-
      group_by(annotations$data$tokens[[text_index()]], sentence_idx) %>%
      group_split()
    rankings <-
      annotations$data$sentences[[text_index()]]$ranking
    value <-
      annotations$data$paragraphs$rating[[text_index()]]
    word_index <- 0
    sentences <- lapply(seq_along(words_by_sentence), function(i) {
      words <- words_by_sentence[[i]]$tokens
      word_annotations <- words_by_sentence[[i]]$annotation
      tags$div(
        style = "display: flex; margin: 10px; justify-content: space-between;",
        tags$div(
          style = "flex: left;",
          lapply(seq_along(words), function(j) {
            word_index <<- word_index + 1
            color <-
              if (word_annotations[j] == 1) "red"
              else if (word_annotations[j] == -1) "green"
              else "black"
            tags$span(words[j], " ", id = word_index,
              class = "word", `data-annotation` = word_annotations[j],
              style = glue("cursor: pointer; font-size: 20px; 
                        user-select: none; color: {color};")
            )
          }),
        ),
        tags$div(
          style = "flex: right; max-width: 120px;",
          selectInput(paste0("sentence_", i), paste("Rank Sentence:"),
            choices = 10:-10,
            selected = rankings[i]
          )
        )
      )
    })
    tags$div(
      tags$h3(
        style = "margin-bottom: 20px;",
        paste0("Paragraph ", text_index(), " / ", length(texts$tokens))
      ),
      sentences,
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
  tags$h1("Text Highlighting tool - Instructions"),
  
  tags$h2("Purpose"),
  tags$p("In order to evaluate how well different models “highlights” words and sentences, we need a dataset to compare against. This dataset will consist of words that are highlighted based on their contribution to the passage rating, so words or sequence of words that contribute to a higher score i.e. depression will be highlighted red and words that contribute to a lower score i.e. healthy will be highlighted green. The entire passage will also be rated, and each sentence will be ranked based on contribution towards depression. With this dataset we can then compare how the model has highlighted words and sentences, and how it has rated the passage to how experts would, providing us with a good metric to evaluate different models."),
  
  tags$h2("Instructions"),
  tags$ul(
    tags$li(
      tags$p(
        tags$strong("Read the passage:"),
        " Before marking or rating, start by reading the full paragraph from start to finish."
      )
    ),
    tags$li(
      tags$p(
        tags$strong("Rate the passage:"),
        " Try, to the best of your ability, to estimate the rating scale score (PHQ-9 : from 0-27) of the participant, based on the text response you just read."
      )
    ),
    tags$li(
      tags$p(
        tags$strong("Marking words:"),
        " If you think a word or sequence of words, accounting for the context in which they are used, is contributing to depression or good mental health, you can click on each word to mark them. If you click it once it will become red, marking it as depressive; if you click it twice it will become green, marking it as healthy; and if you click it again it will go back to being neutral. \n
        Please highlight words/sequences accordingly: ",
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
        tags$strong("Ranking sentences:"), 
        " Next to each sentence is a dropdown menu. Use this menu to rank the sentences in order of how much they contribute to a higher or lower depression score. The scale here is 10 to -10 where 10 means it’s an extremely depressive sentence, 0 means it’s neutral or that it doesn’t give any information, and -10 means that the sentence indicates very good mental health."
      )
    ),
    tags$li(
      tags$p(
        tags$strong("Technical details:"), 
        " There is a counter on the top of the page that tells you how many paragraphs you have highlighted. Whenever you click 'Next Text', 'Previous Text' or 'Save Annotations' your selections will be saved locally on your browser. This means if you ever close or update the webpage, your progress will be saved for all paragraphs except the one you are currently working on. When you are done, press the 'Save annotations' button, save the file as 'Annotations-<YourName>.rds' and let us know that you are done."
      )
    ),
  ),
    tags$p("If you have any questions, don’t hesitate to reach out."),
    tags$p(
      tags$strong("Thank you so much for helping out!")
    ),
  
  tags$h2("Question and instructions given to the participants:"),
  tags$p("Over the last 2 weeks, have you been depressed or not? Please answer the question by typing at least a paragraph below that indicates whether you have been depressed or not. Try to weigh the strength and the number of aspects that describe if you have been depressed or not so that they reflect your overall personal state of depression. For example, if you have been depressed, then write more about aspects describing this, and if you have not been depressed, then write more about aspects describing that."),
  tags$p("Write about those aspects that are most important and meaningful to you."),
  tags$p("Write at least one paragraph in the box.")
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
        "Sentence Ranking Legend"
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
    annotations$data$tokens <- lapply(input$restored_annotations$tokens,
                                      as_tibble)
    if (!is.null(annotations$data$text_index)) {
      text_index(annotations$data$text_index)
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
    annotations$data$tokens[[text_index()]]$annotation <-
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
