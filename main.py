from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import evaluate
import accelerate
import transformers
import numpy as np
import optuna
import random
from datasets import load_dataset

print('finished imports')

device = 'mps'
# Set device to cuda/mps/cpu
#if torch.cuda.is_available():
    #device = "cuda"
#if torch.backends.mps.is_available():
    #device = "mps"
#print(device)

#evaluation
metric = evaluate.load("accuracy")


def get_dataset(path:str, num_shards=5000):
    '''Loads dataset from HuggingFace, returns a single shard
    '''
    dataset = load_dataset(path)

    split_name = list(dataset.keys())[0]
    sub_dataset = dataset[split_name].shard(num_shards, 1)

    return sub_dataset

    

def tokenize(dataset, tokenizer:AutoTokenizer, example_index:str):
    '''tokenize, pads, and truncates dataset object'''
    dataset = dataset.map(lambda examples: tokenizer(examples[example_index],
                                                   return_tensors="pt",
                                                   padding=True, truncation=True),
                        batched=True).with_format("torch")
    return dataset

def split_dataset(dataset, train_size:float, test_size:float, eval_size:float):
    train_set, test_set, eval_set = torch.utils.data.random_split(dataset, [train_size, test_size, eval_size])
    return train_set, test_set, eval_set

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

dataset = get_dataset('dliu1/legal-llama-raw-text')

tokenized_dataset = tokenize(dataset, tokenizer=tokenizer, example_index='text')
print(tokenized_dataset[2]) #prints tokenized tensor of one entry

train_data, test_data, eval_data = split_dataset(dataset, 0.8, 0.1, 0.1)

print(len(train_data))
print(len(test_data))
print(len(eval_data))
print('finished')

model = AutoModelForSequenceClassification.from_pretrained(model_name, load_in_8bit=True).to(device)

args = TrainingArguments(
        f"{model_name}-RE_Llama",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=True
)

trainer = Trainer(
    model = model,
    args=args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

for obj in trainer.state.log_history:
    print(obj)
