import subprocess
import json
import sys

# Read the JSON file
with open(sys.argv[1]) as json_file:
    data = json.load(json_file)

# Parse the content of the JSON file as command line arguments
arg_list = [f' --{key} {value}' for key, value in data.items()]
args = ''.join(arg_list)

# Run the model
subprocess.run(f'CUDA_VISIBLE_DEVICES=0 python3 train_bert.py {args}', shell=True)