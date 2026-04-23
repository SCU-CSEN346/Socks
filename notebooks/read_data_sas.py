import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

raw_dataset = pd.read_csv("../data/train_rel_2.tsv", sep='\t', encoding='latin1')

def process_essay_set(raw_dataset, ix, drop_cols, seed=42):
    df = raw_dataset[raw_dataset["EssaySet"] == ix].copy()

    actual_drop = list(set(drop_cols + ["Score2"]))
    df = df.drop(columns=actual_drop, errors="ignore")
    
    if "Score1" in df.columns:
        df = df.rename(columns={"Score1": "score"})
    
    df.columns = [c.lower() for c in df.columns]
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    split_index = int(0.8 * len(df))
    df["split"] = "test"
    df.loc[:split_index-1, "split"] = "train"
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

DROP_CONFIG = {i: [] for i in range(1,11)}

output_dir = Path("../data/sas")
output_dir.mkdir(parents=True, exist_ok=True)

for ix in range(1, 11):
    if ix not in raw_dataset["EssaySet"].unique():
        continue
        
    df_processed = process_essay_set(
        raw_dataset,
        ix,
        drop_cols=DROP_CONFIG[ix],
        seed=RANDOM_SEED
    )
    
    output_path = output_dir / f"essay_set_{ix}_sas.csv"
    df_processed.to_csv(output_path, index=False)
    
    print(f"Saved: {output_path}")