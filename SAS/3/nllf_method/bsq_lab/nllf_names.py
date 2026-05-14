import pandas as pd
import json
from pathlib import Path

Path("output/").mkdir(parents=True, exist_ok=True)

bsqs = pd.read_csv("output/clustered_bsqs.csv", index_col=0)
bsqs["id"] = bsqs["cluster"].apply(lambda x: f"c{x}")
bsqs = bsqs[bsqs["is_centroid"]].reset_index().set_index("id")["bsq"].to_dict()
print(bsqs)

named_nllf = {
    f"{k} ({a})": f"{v} Yes" if a == "Y" else f"{v} No"
    for k, v in bsqs.items()
    for a in "YN"
}

transformed_names = {
    f"{k} ({a})": (
        v.replace("Does the answer ", "The answer ").rstrip("?") + "."
        if a == "Y"
        else v.replace("Does the answer ", "The answer does not ").rstrip("?") + "."
    )
    for k, v in bsqs.items()
    for a in "YN"
}
print(transformed_names)

with open("output/transformed_names.json", "w") as json_file:
    json.dump(transformed_names, json_file, indent=4)

print("transformed_names.json saved to output/transformed_names.json")
