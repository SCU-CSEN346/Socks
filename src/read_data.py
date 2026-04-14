import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

raw_dataset = pd.read_csv("../data/training_set_rel3.tsv", sep='\t', encoding='latin1')

def process_essay_set(raw_dataset, ix, drop_cols, rename_score=True, seed=42):
    df = raw_dataset[raw_dataset["essay_set"] == ix].copy()
    df = df.dropna(axis=1)
    df = df.drop(columns=drop_cols, errors="ignore")
    
    if rename_score and "domain1_score" in df.columns:
        df = df.rename(columns={"domain1_score": "score"})
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    split_index = int(0.8 * len(df))
    df["split"] = ["train"] * split_index + ["test"] * (len(df) - split_index)
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

DROP_CONFIG = {
    1: "rater1_domain1 rater2_domain1".split(),
    2: "rater1_domain1 rater2_domain1 rater1_domain2 rater2_domain2".split(),
    3: "rater1_domain1 rater2_domain1".split(),
    4: "rater1_domain1 rater2_domain1".split(),
    5: "rater1_domain1 rater2_domain1".split(),
    6: "rater1_domain1 rater2_domain1".split(),
    7: (
        "rater1_domain1 rater2_domain1 "
        "rater1_trait1 rater1_trait2 rater1_trait3 rater1_trait4 "
        "rater2_trait1 rater2_trait2 rater2_trait3 rater2_trait4"
    ).split(),
    8: (
        "rater1_domain1 rater2_domain1 "
        "rater1_trait1 rater1_trait2 rater1_trait3 rater1_trait4 rater1_trait5 rater1_trait6 "
        "rater2_trait1 rater2_trait2 rater2_trait3 rater2_trait4 rater2_trait5 rater2_trait6"
    ).split(),
}

output_dir = Path("../data")
output_dir.mkdir(parents=True, exist_ok=True)

for ix in range(1, 9):
    df_processed = process_essay_set(
        raw_dataset,
        ix,
        drop_cols=DROP_CONFIG[ix],
        seed=RANDOM_SEED
    )
    
    output_path = output_dir / f"essay_set_{ix}.csv"
    df_processed.to_csv(output_path)
    
    print(f"Saved: {output_path}")

