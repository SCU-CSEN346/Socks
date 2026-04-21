INDEX = 0
assert INDEX in [0,1]
CUDA_DEVICE = f"cuda:{INDEX}"

import json
PARAMS = json.load(open("../params.json", "r"))

LLM = PARAMS["LLM"]
ESSAY_SET = PARAMS["ESSAY_SET"]

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

first_ex = lambda q, a: lambda_instance.replace("[[A]]", q).replace("[[E]]",  a)
template = lambda init_I, init_A, q, a: f"""<s> [INST] {init_I} [/INST] {init_A} </s> [INST] {first_ex(q, a)} [/INST]"""

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

with open('../../../data/question.json', 'r') as f:
    meta_type = json.load(f)

df_QA = pd.read_csv(f"../../../data/essay_set_{ESSAY_SET}.csv", index_col=0)
df_QA = df_QA[df_QA["split"] == "train"]

sub_df_index = df_QA.index[:df_QA.shape[0]//2] if '0' in CUDA_DEVICE else df_QA.index[df_QA.shape[0]//2:]
responses = []
for i, ix in enumerate(sub_df_index):
    r = df_QA.loc[ix]["essay"]
    p = meta_type[str(ESSAY_SET)]["question"]
    o = llm_call(p, r)
    o = o.replace("\n", " ").strip()
    responses.append([ix, o])
    print(p, r)
    print(100*(i+1)/sub_df_index.shape[0], ix, o)

sample = df_QA.loc[sub_df_index].copy()
sample["llm"] = [x[1] for x in responses]

sample.to_csv(f"data/train_{CUDA_DEVICE.replace(':', '_')}.csv")