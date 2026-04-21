CUDA_DEVICE = "cuda:0"

import json
PARAMS = json.load(open("../params.json", "r"))

SEED = PARAMS["SEED"]
NUM_EX = PARAMS["NUM_EX"]
LLM = PARAMS["LLM"]

from pathlib import Path
import pandas as pd
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

directory = "output/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory output/ created successfully!")

bsq_data_train = pd.read_csv("data/bsq_data_train.csv", index_col=0)
random_examples = bsq_data_train.sample(NUM_EX, random_state=SEED)

with open('bsq_gen_prompt.json', 'r') as f:
    prompt_details = json.load(f)

initial_message = prompt_details["rol"]
initial_response = prompt_details["check"]
lambda_instance = prompt_details["instance"]

first_ex = lambda p, r: lambda_instance.replace("[[P]]", p).replace("[[R]]",  r)
template = lambda init_I, init_A, p, r: f"""<s> [INST] {init_I} [/INST] {init_A} </s> [INST] {first_ex(p, r)} [/INST]"""

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

    inputs = tokenizer(prompt, return_tensors="pt").to(0)

    outputs = model.generate(**inputs, max_new_tokens=500, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    o = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return o.split("[/INST]")[-1].strip()

p = random_examples["Q"].iloc[0]
r = random_examples["A"].iloc[0]

responses = []
for i, ix in enumerate(random_examples.index[:]):
    p, r = random_examples.loc[ix]["Q A".split()]
    o = llm_call(p, r)
    o = o.replace("\n", " ").strip()
    responses.append([ix, o])
    print(p, r)
    print(100*(i+1)/random_examples.shape[0], ix, o)
    
sample = random_examples.loc[random_examples.index[:]].copy()
sample["llm"] = [x[1] for x in responses]

sample.to_csv("output/sample.csv", index=0)

print("BSQ were generated successfully!")