import json
import time

PARAMS = json.load(open("../params.json", "r"))

LLM = PARAMS["LLM"]
ESSAY_SET = PARAMS["ESSAY_SET"]
MISTRAL_API_KEY = PARAMS.get("MISTRAL_API_KEY", "ZFDoX8XITaNFoHxgn1SabB9dDOOvY5Cj")

from pathlib import Path
import pandas as pd
from mistralai.client import Mistral

directory = "labels/"
path = Path(directory)
path.mkdir(parents=True, exist_ok=True)
print("Directory labels/ created successfully!")

bsq_data_train = pd.read_csv("data/bsq_data_train.csv")
clustered_bsqs = pd.read_csv("output/clustered_bsqs.csv")

bsqs = {}
for c in clustered_bsqs[clustered_bsqs["is_centroid"]].cluster:
    bsqs[f"c{c}"] = clustered_bsqs[
        (clustered_bsqs["is_centroid"]) & (clustered_bsqs["cluster"] == c)
    ].iloc[0].bsq

print(f"Total BSQ clusters: {len(bsqs)}")
print(f"Cluster keys: {list(bsqs.keys())}")

with open('weak_label_prompt.json', 'r') as f:
    prompt_details = json.load(f)

initial_message = prompt_details["rol"]
initial_response = prompt_details["check"]
lambda_instance = prompt_details["instance"]

client = Mistral(api_key=MISTRAL_API_KEY)
MODEL_NAME = "mistral-small-2506"

def prep(text):
    return str(text).replace("\n", " ").strip()

def llm_call(q, a, b):
    p = prep(q)
    r = prep(a)
    s = prep(b)

    user_prompt = lambda_instance.replace("[[Q]]", p).replace("[[E]]", r).replace("[[B]]", s)

    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {"role": "system",    "content": initial_message},
            {"role": "assistant", "content": initial_response},
            {"role": "user",      "content": user_prompt},
        ],
        max_tokens=5,
        temperature=0,
    )

    return response.choices[0].message.content.strip()

with open('../../../data/question.json', 'r') as f:
    meta_type = json.load(f)

all_bsqs = list(bsqs.items())

for key, b in all_bsqs:
    print(f"\n--- Processing {key} ---")
    print(f"BSQ: {b}")
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
    print(f"Saved labels/llm_responses_{key}.xlsx")

print("\nAll BSQ clusters labelled successfully!")