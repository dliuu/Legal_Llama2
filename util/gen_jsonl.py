import json
import re

filename = "raw_txt_files/real_property.txt"
output = 'real_property.jsonl'

# Generate a list of dictionaries
lines = []
with open(filename, encoding="utf8") as f:
    for line in f.read().splitlines():
        if line:
            #clean text
            cleaned_line = line.replace('\t', "")
            cleaned_line = cleaned_line.replace('/', "")
            cleaned_line = cleaned_line.replace('"', "")
            lines.append({"text": cleaned_line})

# Convert to a list of JSON strings
json_lines = [json.dumps(l) for l in lines]

# Join lines and save to .jsonl file
json_data = '\n'.join(json_lines)
with open(output, 'w') as f:
    f.write(json_data)