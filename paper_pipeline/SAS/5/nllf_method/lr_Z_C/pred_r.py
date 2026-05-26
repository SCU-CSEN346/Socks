import json
PARAMS = json.load(open("params.json", "r"))

SEED = PARAMS["SEED"]
RANDOM_SEED = PARAMS["RANDOM_SEED"]
S_MAX = PARAMS["S_MAX"]
S_MIN = PARAMS["S_MIN"]
CV = PARAMS["CV"]

ESSAY_SET = PARAMS["ESSAY_SET"]

import pandas as pd
import numpy as np
import random
from sklearn_genetic import GAFeatureSelectionCV

GAFeatureSelectionCVWeighted = GAFeatureSelectionCV

from sklearn.linear_model import Ridge
from pathlib import Path
from sklearn.preprocessing import StandardScaler

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

SIGNAL_PATH = "/llm/weak_signal/data"
WITH_SIGNAL_FILTERING = True

foo = "w_r" if WITH_SIGNAL_FILTERING else "wo"
bar = "ulra"

if "llm" in SIGNAL_PATH:
    bar = "llm"
elif "signal_clustering" in SIGNAL_PATH:
    bar = "signal_clustering"

for directory in ["pred/", f"pred/{bar}/", f"pred/{bar}/{foo}_sf/", f"pred/{bar}/{foo}_sf/nllf/", f"pred/{bar}/{foo}_sf/nllf_ef/", f"pred/{bar}/{foo}_sf/ef/"]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory pred/ created successfully!")

if WITH_SIGNAL_FILTERING:
    gt_1_name = "../../signal_clustering/weak_signal/data/train.csv"
    if bar == "llm":
        gt_1_name = "../../llm/weak_signal/data/train.csv"

    df_gt_1 = pd.read_csv(gt_1_name, index_col=0)
    df_gt_2 = pd.read_csv("../../ulra_paper/weak_signal/data/train.csv", index_col=0)

    df_gt = df_gt_1.copy()
    df_gt["other_pred"] = df_gt_2["pred"]

    def score2label(score, res=0, size=7, m=0.49, M=1.69):
        step = (M-m) / (size)
        if score < step + m:
            return 1+res
        else:
            return score2label(score-step, res+1, size, m, M)

    y_real_2 = df_gt["other_pred"]
    m, M = np.percentile(y_real_2, 1), np.percentile(y_real_2, 99)
    preds = y_real_2.apply(lambda x: score2label(x, res=S_MIN-1, size=S_MAX-S_MIN+1, m=m, M=M)).values
    preds = np.array([p if S_MIN <= p and p <= S_MAX else S_MIN if p < S_MIN else S_MAX for p in preds])
    df_gt["other_pred_quantize"] = preds

    if bar in ["signal_clustering", "ulra"]:
        y_real_1 = df_gt["pred"]
        m, M = np.percentile(y_real_1, 1), np.percentile(y_real_1, 99)
        preds = y_real_1.apply(lambda x: score2label(x, res=S_MIN-1, size=S_MAX-S_MIN+1, m=m, M=M)).values
        preds = np.array([p if S_MIN <= p and p <= S_MAX else S_MIN if p < S_MIN else S_MAX for p in preds])
        df_gt["pred"] = preds

    signal_filtering = 1 - ((df_gt["pred"] - df_gt["other_pred_quantize"]).abs() / (S_MAX-S_MIN))
    print(SIGNAL_PATH, foo, bar, signal_filtering.min(), signal_filtering.max())

df_gt = pd.read_csv(f"../..{SIGNAL_PATH}/train.csv", index_col=0)

df = {}
for k in ["train", "test"]:
    o = pd.read_csv(f"../nllf/labels/data_nllf.csv", index_col=0)
    df[k] = o[o["split"] == k]
    if "train" in k:
        df[k]["score"] = df_gt.loc[df[k].index]["pred"]

    o = pd.read_csv(f"../../ulra_paper/lf/data/data.csv", index_col=0)
    u = o[o["split"] == k].drop(columns="index split".split())
    u.columns = [f"ulra_{c}" for c in u.columns]
    df[k] = pd.concat([df[k], u], axis=1)

print(df["train"].info())

for FEATURE_SET in ["nllf", "ef", "nllf_ef"]:

    nllf_features_name = sorted([c for c in df["train"].columns if all(bool(e in c) for e in "c)(")])
    ulra_features_name = sorted([c for c in df["train"].columns if "ulra_" in c])

    features_name = None
    if FEATURE_SET == "nllf":
        features_name = nllf_features_name
    elif FEATURE_SET == "ef":
        features_name = ulra_features_name
    elif FEATURE_SET == "nllf_ef":
        features_name = nllf_features_name + ulra_features_name

    X_train = df["train"][features_name]
    common_cols = sorted(list(X_train.columns))
    y_train = df["train"]["score"]
    train_sample = pd.concat([X_train, y_train], axis=1)

    evolved_estimator = GAFeatureSelectionCV(
        estimator=Ridge(alpha=10),
        cv=CV,
        scoring="r2",
        population_size=30,
        generations=30,
        n_jobs=-1,
        verbose=True,
        keep_top_k=2,
        elitism=True,
    )

    corr = json.load(open("../bsq_lab/output/corr.json", "r"))
    c_selected_features = [c for c in common_cols if c in nllf_features_name and corr[c]>0]
    support = sorted([x for x in X_train.columns if x in c_selected_features or x in ulra_features_name])

    std_scaler = StandardScaler()

    evolved_estimator.fit(
        pd.DataFrame(std_scaler.fit_transform(train_sample.drop(columns="score")[support])).values,
        train_sample["score"]
    )

    best_features = [x for i, x in enumerate(train_sample.drop(columns="score")[support].columns) if vars(evolved_estimator)["best_features_"][i]]
    new_best_features = best_features

    with open(f"pred/{bar}/{foo}_sf/{FEATURE_SET}/best_features.npy", 'wb') as f:
        np.save(f, np.array(new_best_features))

    clf = Ridge(alpha=10)
    std_scaler = StandardScaler()

    if WITH_SIGNAL_FILTERING:
        clf.fit(std_scaler.fit_transform(X_train[new_best_features]), y_train, sample_weight=signal_filtering)
    else:
        clf.fit(std_scaler.fit_transform(X_train[new_best_features]), y_train)

    X_test = df["test"][common_cols]
    y_pred = clf.predict(std_scaler.transform(X_test[new_best_features]))

    df_test = df["test"].copy()[["index", "Id"]]
    df_test["pred"] = y_pred
    df_test.to_csv(f"pred/{bar}/{foo}_sf/{FEATURE_SET}/test.csv")

    print(f"Predictions saved to pred/{bar}/{foo}_sf/{FEATURE_SET}/test.csv")
