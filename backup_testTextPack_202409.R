# Issue https://github.com/OscarKjell/text/issues/192

#reticulate::source_python("./huggingface_Interface3.py"
#)

.rs.restartR()
install.packages(c("devtools", "reticulate"))
unlink(reticulate::miniconda_path(), recursive = TRUE, force = TRUE)
file.exists(reticulate::miniconda_path())

install.packages('text') # No need to install conda 
#devtools::install_github("oscarkjell/text") # will need conda installed beforehand
#devtools::install_github('moomoofarm1/textPlot@textInitialize')
text::textrpp_install() # Will install conda + python env.

.rs.restartR()
reticulate::use_condaenv(condaenv = "textrpp_condaenv")
reticulate::py_run_string(
  'import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import torch
import huggingface_hub
from transformers import AutoConfig, AutoModel, AutoTokenizer
try:
    from transformers.utils import logging
except ImportError:
    print("Warning: Unable to importing transformers.utils logging")
from transformers import pipeline
import numpy as np

import nltk
try:
    nltk.data.find("tokenizers/punkt/PY3/english.pickle")
except:
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize

import os, sys'
)


text::textrpp_initialize(save_profile = TRUE)





# ===== END =====

reticulate::install_miniconda(force=TRUE)
reticulate::conda_create(envname="textrpp_condaenv", python_version="3.9")
reticulate::conda_install(envname="textrpp_condaenv", packages=c( "torch==2.2.0", "flair==0.13.0" ), pip=TRUE)
rpp_version <- c("nltk==3.6.7")
reticulate::conda_install(envname="textrpp_condaenv", packages=rpp_version)
devtools::install_github('moomoofarm1/textPlot@textInitialize')
