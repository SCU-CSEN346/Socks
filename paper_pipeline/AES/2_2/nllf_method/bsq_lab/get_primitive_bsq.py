CUDA_DEVICE = "cuda:0"

import json
PARAMS = json.load(open("../params.json", "r"))

SEED = PARAMS["SEED"]
LLM = PARAMS["LLM"]
HF_TOKEN = PARAMS["HF_TOKEN"]
ESSAY_SET = PARAMS["ESSAY_SET"]
DOMAIN = PARAMS["DOMAIN"]

from pathlib import Path
import pandas as pd
import json
from mistralai.client import Mistral   # pip install mistralai

directory = "output/"
path = Path(directory)
path.mkdir(parents=True, exist_ok=True)
print("Directory output/ created successfully!")

with open('bsq_gen_prompt.json', 'r') as f:
    prompt_details = json.load(f)

# ---- Mistral API client (replaces AutoModelForCausalLM) ----
MISTRAL_API_KEY = PARAMS.get("MISTRAL_API_KEY", "ZFDoX8XITaNFoHxgn1SabB9dDOOvY5Cj")
client = Mistral(api_key=MISTRAL_API_KEY)
MODEL_NAME = "mistral-small-2506"  # or "open-mixtral-8x7b" to match original
# ------------------------------------------------------------

def prep(text):
    return str(text).replace("\n", " ").strip()

def llm_call(text, type_template):
    text = prep(text)

    initial_message = prompt_details[type_template]["rol"]
    initial_response = prompt_details[type_template]["check"]
    lambda_instance = prompt_details[type_template]["instance"]

    user_prompt = lambda_instance.replace("[[P]]", text)

    # Replaces the [INST] template + model.generate() block
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {"role": "system",    "content": initial_message},
            {"role": "assistant", "content": initial_response},  # preserves few-shot primer
            {"role": "user",      "content": user_prompt},
        ],
        max_tokens=500,
        temperature=0,   # equivalent to do_sample=False
    )

    return response.choices[0].message.content.strip()


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
        for c, u in v[str(DOMAIN)].items():
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