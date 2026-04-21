INDEX = 1
assert INDEX in [0,1]

CUDA_DEVICE = f"cuda:{INDEX}"

import json
PARAMS = json.load(open("../params.json", "r"))

LLM = PARAMS["LLM"]
ESSAY_SET = PARAMS["ESSAY_SET"]

from pathlib import Path

directory = "labels/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory labels/ created successfully!")

import pandas as pd

bsq_data_train = pd.read_csv("data/bsq_data_train.csv")

clustered_bsqs = pd.read_csv("output/clustered_bsqs.csv")

bsqs = {}
for c in clustered_bsqs[clustered_bsqs["is_centroid"]].cluster:
    bsqs[f"c{c}"] = clustered_bsqs[(clustered_bsqs["is_centroid"]) & (clustered_bsqs["cluster"] == c)].iloc[0].bsq

import json
with open('weak_label_prompt.json', 'r') as f:
    prompt_details = json.load(f)

initial_message = prompt_details["rol"]
initial_response = prompt_details["check"]
lambda_instance = prompt_details["instance"]

first_ex = lambda q, a, b: lambda_instance.replace("[[Q]]", q).replace("[[E]]",  a).replace("[[B]]",  b)

template = lambda init_I, init_A, q, a, b: f"""<s> [INST] {init_I} [/INST] {init_A} </s> [INST] {first_ex(q, a, b)} [/INST]"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.set_device(CUDA_DEVICE)

model_id = LLM
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

import gc
torch.cuda.empty_cache()
gc.collect()

def prep(text):
    return str(text).replace("\n", " ").strip()

def llm_call(q, a, b):

    p = prep(q)
    r = prep(a)
    s = prep(b)
    
    prompt = template(initial_message, initial_response, p, r, s)

    inputs = tokenizer(prompt, return_tensors="pt").to(CUDA_DEVICE)

    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    o = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return o.split("[/INST]")[-1].strip()

sub_bsqs = list(bsqs.items())[:len(bsqs) // 2] if '0' in CUDA_DEVICE else list(bsqs.items())[len(bsqs) // 2:]

with open('../../../data/question.json', 'r') as f:
    meta_type = json.load(f)

for key, b in sub_bsqs:
    print(key, b)
    responses = []
    for i, ix in enumerate(bsq_data_train.index):
        r = bsq_data_train.loc[ix]["essay"]
        p = meta_type[str(ESSAY_SET)]["question"]
        o = llm_call(p, r, b)
        o = o.replace("\n", " ").strip()
        responses.append([ix, o])
        print(f"{key}: {100*(i+1)/bsq_data_train.shape[0]:.2f}%", ix, o)

    train_bi = bsq_data_train.copy()
    train_bi["llm"] = [x[1] for x in responses]

    train_bi.to_excel(f"labels/llm_responses_{key}.xlsx")
