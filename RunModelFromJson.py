import subprocess
import json
import sys

# Read the JSON file

input_args = sys.argv

with open(input_args[1]) as json_file:
    data = json.load(json_file)

isBert = input_args[2] == "bert"

# Parse the content of the JSON file as command line arguments
arg_list = [f' --{key} {value}' for key, value in data.items()]
args = ''.join(arg_list)

fname = "train_bert.py" if isBert else "train.py"

# Run the model
subprocess.run(f'CUDA_VISIBLE_DEVICES=0 python3 {fname} {args}', shell=True)