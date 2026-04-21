INDEX = 0
assert INDEX in [0,1]
CUDA_DEVICE = f"cuda:{INDEX}"

import json
PARAMS = json.load(open("../params.json", "r"))

LLM = PARAMS["LLM"]

from pathlib import Path

directory = "data/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory data/ created successfully!")

import pandas as pd
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

with open('prompt.json', 'r') as f:
    prompt_details = json.load(f)

initial_message = prompt_details["rol"]
initial_response = prompt_details["check"]
lambda_instance = prompt_details["instance"]

first_ex = lambda p, r: lambda_instance.replace("[[P]]", p).replace("[[R]]",  r)

template = lambda init_I, init_A, p, r: f"""<s> [INST] {init_I} [/INST] {init_A} </s> [INST] {first_ex(p, r)} [/INST]"""

print(torch.cuda.is_available())

torch.cuda.set_device(CUDA_DEVICE)

model_id = LLM
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

torch.cuda.empty_cache()
gc.collect()

def prep(text):
    return str(text).replace("\n", " ").strip()

def llm_call(q, a):

    p = prep(q)
    r = prep(a)

    prompt = template(initial_message, initial_response, p, r)

    inputs = tokenizer(prompt, return_tensors="pt").to(CUDA_DEVICE)

    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    o = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return o.split("[/INST]")[-1].strip()

df_QA = pd.read_csv("../../data/train_val.csv", index_col=0)
sub_df_index = df_QA.index[:df_QA.shape[0]//2] if '0' in CUDA_DEVICE else df_QA.index[df_QA.shape[0]//2:]

responses = []
for i, ix in enumerate(sub_df_index):
    p, r = df_QA.loc[ix]["Q  A".split("  ")]
    o = llm_call(p, r)
    o = o.replace("\n", " ").strip()
    responses.append([ix, o])
    print(p, r)
    print(2*100*(i+1)/df_QA.shape[0], ix, o)
sample = df_QA.loc[sub_df_index].copy()[["index"]]
sample["llm"] = [x[1] for x in responses]

sample.to_csv(f"data/{CUDA_DEVICE.replace(':', '_')}.csv")