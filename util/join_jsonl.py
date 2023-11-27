'''
Appends JSONL Files
'''

import json

def append_jsonl_files(input_files, output_file):
    with open(output_file, 'a', encoding='utf-8') as output_jsonl:
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as input_jsonl:
                for line in input_jsonl:
                    output_jsonl.write(line)

# Variables
input_files = ['jsonl/proceedings.jsonl', 'jsonl/PropertyTax.jsonl', 'jsonl/real_property.jsonl']
output_file = 'RawText_Set.jsonl'

append_jsonl_files(input_files, output_file)
