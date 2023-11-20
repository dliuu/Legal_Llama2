from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import evaluate
import accelerate
import transformers
import numpy as np
import optuna
import random
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

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

#evaluation
metric = evaluate.load("accuracy")


def get_dataset(path:str, tokenizer:AutoTokenizer):
    '''Loads dataset, passes it into a tokenizer, pads and truncate data
    '''
    dataset = load_dataset(path)
    random.shuffle(dataset)
    dataset = dataset.map(lambda examples: tokenizer(examples["text"],
                                                   return_tensors="pt",
                                                   padding=True, truncation=True),
                        batched=True).with_format("torch")
    return dataset
    


def model_init():
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#Main:
model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

dataset = get_dataset('dliu1/re_propertytax', tokenizer)
print(dataset['train'][10])

train_small_dataset = get_dataset('dliu1/re_propertytax', tokenizer, percent=0.05)
eval_dataset = get_dataset('dliu1/re_propertytax', tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

args = TrainingArguments(
        f"{model_name}-RE_Llama",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=2,
        weight_decay=0.01
)

trainer = Trainer(
    model = model,
    args=args,
    train_dataset=train_small_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)