import json
PARAMS = json.load(open("../params.json", "r"))

S_MAX = PARAMS["S_MAX"]

from pathlib import Path

directory = "pred/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory pred/ created successfully!")

def retrive_score(x):
    pred = (1 + S_MAX) // 2
    hint = [
        "Nota: ", 
        "la respuesta es un",
        "nota de la respuesta sería un", 
        "nota de la respuesta es un",
        "nota de la respuesta es",
        "nota de",
        "una nota",
        "sería un",
        "una nota",
        "nota es un",
        "una nota de",
        "La nota sería",
        "respuesta es un",
        "alumno es un",
    ]
    s = True
    i = 0
    while s:
        if hint[i] in x:
            h = hint[i]
            try:
                pred = int(x[x.index(h)+len(h):].strip()[0])
                s = False
            except:
                i+1
        else:
            i+=1

    h = hint[i]
    pred = int(x[x.index(h)+len(h):].strip()[0])
    
    if pred > S_MAX:
        return S_MAX
    elif pred < 1:
        return 1
    else:
        return pred
    
import pandas as pd

df_0 = pd.read_csv(f"data/cuda_0.csv", index_col=0)
df_1 = pd.read_csv(f"data/cuda_1.csv", index_col=0)

df = pd.concat([df_0, df_1])

df["pred"] = df["llm"].apply(retrive_score)

df_test = df.copy()["index pred".split()]

df_test.to_csv(f"pred/test.csv")
