#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install optuna transformers datasets accelerate evaluate')


# In[ ]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, evaluate, accelerate
from transformers import TrainingArguments, Trainer
import numpy as np
import optuna
import random


# In[ ]:


# Set device to cuda/mps/cpu
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
device


# In[ ]:


modelname = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(modelname,
                                                            num_labels=2).to(device)

