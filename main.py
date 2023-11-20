from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import evaluate
import accelerate
import transformers
import numpy as np
import optuna
import random

#from petals import AutoDistributedModelForCausalLM
#from huggingface_hub import login
#login("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")

print('finished imports')

# Set device to cuda/mps/cpu
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(device)

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
