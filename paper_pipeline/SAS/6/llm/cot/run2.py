INDEX = 0
assert INDEX in [0,1]
CUDA_DEVICE = f"cuda:{INDEX}"

import time
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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
from mistralai.client import Mistral
import gc

with open('prompt.json', 'r') as f:
    prompt_details = json.load(f)

initial_message = prompt_details["rol"]
initial_response = prompt_details["check"]
lambda_instance = prompt_details["instance"]

first_ex = lambda q, a: lambda_instance.replace("[[A]]", q).replace("[[E]]", a)
template = lambda init_I, init_A, q, a: f"""<s> [INST] {init_I} [/INST] {init_A} </s> [INST] {first_ex(q, a)} [/INST]"""

client = Mistral(api_key="6rtMWIAZu6NIDUMhTK6xKtK3j7B6bBvX")
api_model = "mistral-small-2506"
print(f"Using model: {LLM} via Mistral API as {api_model}")

def prep(text):
    return str(text).replace(chr(10), " ").strip()

def llm_call(q, a):
    p = prep(q)
    r = prep(a)
    prompt = template(initial_message, initial_response, p, r)

    for attempt in range(5):
        try:
            response = client.chat.complete(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            o = response.choices[0].message.content.strip()
            o = o.replace("**", "")
            return o
        except Exception as e:
            if attempt < 4:
                wait = (2 ** attempt) * 15
                print(f"API error (attempt {attempt+1}): {e}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"API error: giving up after 5 attempts: {e}", flush=True)
                return ""

with open('../../../data/question.json', 'r', encoding='utf-8') as f:
    meta_type = json.load(f)

df_QA = pd.read_csv(f"../../../data/essay_set_{ESSAY_SET}.csv", index_col=0)
df_QA = df_QA[df_QA["split"] == "test"]

sub_df_index = df_QA.index
responses = []
for i, ix in enumerate(sub_df_index):
    r = df_QA.loc[ix]["EssayText"]
    p = meta_type[str(ESSAY_SET)]["question"]
    o = llm_call(p, r)
    o = o.replace(chr(10), " ").strip()
    responses.append([ix, o])
    print(p, r)
    print(100*(i+1)/sub_df_index.shape[0], ix, o)

sample = df_QA.loc[sub_df_index].copy()
sample["llm"] = [x[1] for x in responses]

half = len(sample) // 2
sample.iloc[:half].to_csv(f"data/test_cuda_0.csv")
sample.iloc[half:].to_csv(f"data/test_cuda_1.csv")
