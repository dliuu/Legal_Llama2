import openai
import os
import pandas as pd
import time
import re
import json
print('finished imports')

openai.api_key = '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'

def get_completion(role: str, prompt: str, model="gpt-3.5-turbo"):
    '''API call for openai's gpt-3.5-turbo model 
    '''
    messages = [{"role": role, "content": prompt}]

    response = openai.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0,)

    return response.choices[0].message.content

def split_text(text_path:str, chunk_size: int):
    '''splits text into chunks of chunk_size and returns each chunk of text in a list
    inputs: 
        text_path: path of .txt or .csv file
         chunk_size: int of desired chunk size 
    outputs:
        list of text chunks'''
    count = 0
    prompt_str = ''
    return_prompts = []
    file = open(text_path)

    for line in file:
        prompt_str += line
        count += 1
    
        if count % chunk_size == 0:
            return_prompts.append(prompt_str)
            count = 0
            prompt_str = ''
    
    return return_prompts

def completion_to_list(input_text):
    '''converts openai completion to a list of dictionaries of question, answer pairs'''
    qa_pairs = re.split(r'\n\nQ: ', input_text.strip())

    # Remove empty strings resulting from the split
    qa_pairs = [pair for pair in qa_pairs if pair]

    data = []
    for pair in qa_pairs:
        # Split the pair into question and answer
        question, answer = re.split(r'\nA: ', pair)

        #strip all Q: and A: from text
        question = question.strip().lstrip('Q:')
        answer = answer.strip().lstrip('A:')
        
        question = question.strip()
        answer = answer.strip()
        
        #Clean question, answer
        question = question.replace('\t', "")
        question = question.replace('/', "")
        question = question.replace('"', "")
        answer = answer.replace('\t', "")
        answer = answer.replace('/', "")
        answer = answer.replace('"', "")

        qa_dict = {'question': question, 'answer': answer}
        data.append(qa_dict)
    
    return data

def dict_to_jsonl(data:dict, output_file:str):
    ''' writes a python dictionary to a jsonl file '''
    with open(output_file, 'a', encoding='utf-8') as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item) + '\n')





#Main
prompt_list = split_text('raw_txt_files/lora_NYLaw_PropertyTax.txt', 20)
role =  'user'

for idx, prompt in enumerate(prompt_list):
    print('generating response ' + str(idx) + ' ...')
    
    try:
        response = get_completion(role, 'generate question answer pairs for the following text. Please start every question with Q: and every answer with A: ' + str(prompt))
        completion_dict = completion_to_list(response)
        dict_to_jsonl(completion_dict, 'PropertyTax_test.jsonl')

    except Exception as e:
        print('error: ' + str(e))
        continue


    print(response)