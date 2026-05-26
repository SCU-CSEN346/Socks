INDEX = 0
assert INDEX in [0,1]
CUDA_DEVICE = f"cuda:{INDEX}"

import time
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
from mistralai.client import Mistral  # fixed import
import gc

with open('prompt.json', 'r') as f:
    prompt_details = json.load(f)

initial_message = prompt_details["rol"]
initial_response = prompt_details["check"]
lambda_instance = prompt_details["instance"]

first_ex = lambda q, a: lambda_instance.replace("[[A]]", q).replace("[[E]]", a)
template = lambda init_I, init_A, q, a: f"""<s> [INST] {init_I} [/INST] {init_A} </s> [INST] {first_ex(q, a)} [/INST]"""

# Initialize Mistral API client
client = Mistral(api_key="ZFDoX8XITaNFoHxgn1SabB9dDOOvY5Cj")  # replace with your new key after regenerating
api_model = "mistral-small-2506"
print(f"Using model: {LLM} via Mistral API as {api_model}")

def prep(text):
    return str(text).replace("\n", " ").strip()

def llm_call(q, a):
    p = prep(q)
    r = prep(a)
    prompt = template(initial_message, initial_response, p, r)

    response = client.chat.complete(
        model=api_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5
    )

    o = response.choices[0].message.content.strip()
    return o.split("[/INST]")[-1].strip()

with open('../../../data/question.json', 'r', encoding='utf-8') as f:
    meta_type = json.load(f)

df_QA = pd.read_csv(f"../../../data/essay_set_{ESSAY_SET}.csv", index_col=0)
df_QA = df_QA[df_QA["split"] == "test"]

sub_df_index = df_QA.index  # process all data on 1 GPU
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

half = len(sample) // 2
sample.iloc[:half].to_csv(f"data/test_cuda_0.csv")
sample.iloc[half:].to_csv(f"data/test_cuda_1.csv")