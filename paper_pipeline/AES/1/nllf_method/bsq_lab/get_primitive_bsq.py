CUDA_DEVICE = "cuda:0"

import json
PARAMS = json.load(open("../params.json", "r"))

SEED = PARAMS["SEED"]
LLM = PARAMS["LLM"]
HF_TOKEN = PARAMS["HF_TOKEN"]
ESSAY_SET = PARAMS["ESSAY_SET"]

from huggingface_hub import login
login(HF_TOKEN)

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

with open('bsq_gen_prompt.json', 'r') as f:
    prompt_details = json.load(f)

torch.cuda.set_device(CUDA_DEVICE)

model_id = LLM
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

torch.cuda.empty_cache()
gc.collect()

def prep(text):
    return str(text).replace("\n", " ").strip()

def llm_call(text, type_template):

    text = prep(text)

    initial_message = prompt_details[type_template]["rol"]
    initial_response = prompt_details[type_template]["check"]
    lambda_instance = prompt_details[type_template]["instance"]

    first_ex = lambda text: lambda_instance.replace("[[P]]", text)
    template = lambda init_I, init_A, text: f"""<s> [INST] {init_I} [/INST] {init_A} </s> [INST] {first_ex(text)} [/INST]"""
    
    prompt = template(initial_message, initial_response, text)

    inputs = tokenizer(prompt, return_tensors="pt").to(0)

    outputs = model.generate(**inputs, max_new_tokens=500, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    o = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return o.split("[/INST]")[-1].strip()


with open('../../../data/question.json', 'r') as f:
    meta_type = json.load(f)

responses = []
for k, v in meta_type[str(ESSAY_SET)].items():
    if k in ["question"]:
        o = llm_call(v, k)
        o = o.replace("\n", " ").strip()
        responses.append({"source": k, "llm": o})
        print(k, o)
    elif k in ["rubric"]:
        for c, u in v.items():
            for t, w in u.items():
                if t in ["text"]:
                    o = llm_call(w, t)
                    o = o.replace("\n", " ").strip()
                    responses.append({"source": f"{k}_{c}_{t}", "llm": o})
                    print(k, o)
                if t in ["elements"]:
                    for i, e in enumerate(w):
                        o = llm_call(e, t)
                        o = o.replace("\n", " ").strip()
                        responses.append({"source": f"{k}_{c}_{t}_{i}", "llm": o})
                        print(k, o)

    
sample = pd.DataFrame(responses)

sample.to_csv("output/sample.csv", index=0)

print("BSQ were generated successfully!")