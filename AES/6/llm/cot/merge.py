import json
PARAMS = json.load(open("../params.json", "r"))

S_MAX = PARAMS["S_MAX"]
S_MIN = PARAMS["S_MIN"]

import pandas as pd

df_0 = pd.read_csv(f"data/test_cuda_0.csv", index_col=0)
df_1 = pd.read_csv(f"data/test_cuda_1.csv", index_col=0)

df = pd.concat([df_0, df_1])

# other = []
# other_score = []
# for text in df["llm"].values:

def llm_to_score(text):
    score = (S_MAX + S_MIN) / 2
    if "Overall score: " in text:
        p = "Overall score: "
        text = text[text.index(p)+len(p):].strip().replace("/", " ")
        try:
            s = float(text[:2])
            score = s
        except:
            if "score of " in text:
                p = "score of "
                text = text[text.index(p)+len(p):].strip()
                s = float(text[:2])
                score = s
            else:
                print(">", text)
                # break
                pass
    elif "Score: " in text:
        p = "Score: "
        text = text[text.index(p)+len(p):].replace("a", " ").strip().replace(",", " ").replace("/", " ").replace("*", " ").replace(")", " ").replace("-", " ").replace(">", " ").replace("'", " ")
        try:
            s = float(text[:2])
            score = s
        except:
            if "score of " in text:
                p = "score of "
                text = text[text.index(p)+len(p):].strip().replace("*", " ").replace("'", " ")
                try:
                    s = float(text[:2])
                    score = s
                except:
                    if "=" in text:
                        p = "="
                        text = text[text.index(p)+len(p):].strip().replace("*", " ").replace("'", " ")
                        s = float(text[:2])
                        score = s
                    else:
                        print(">>", text)
                        # break
                        pass
            elif "Score: " in text:
                p = "Score: "
                text = text[text.index(p)+len(p):].strip()
                s = float(text[:2])
                score = s
            else:
                print(">>>", text)
                # break
                pass
    elif "a score of" in text:
        p = "a score of"
        text = text[text.index(p)+len(p):].replace("a Grade ", " ").replace("*", " ").replace("Score: ", " ").replace("a", " ").replace(":", " ").strip().replace(",", " ").replace("/", " ")
        s = float(text[:2])
        score = s
    elif "grade of" in text:
        p = "grade of"
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s
    elif "score for the essay is a" in text:
        p = "score for the essay is a"
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s
    elif "score for the essay is " in text:
        p = "score for the essay is "
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s
    elif "Total score: " in text:
        p = "Total score: "        
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s
    elif "final score is " in text:
        p = "final score is "
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s       
    elif "score, such as a " in text:
        p = "score, such as a "
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s       
    elif "score could be a " in text:
        p = "score could be a "
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s     
    elif "score for the essay would be a " in text:
        p = "score for the essay would be a "
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s    
    elif "A score of " in text:
        p = "A score of "
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s   
    elif "score (e.g. a " in text:
        p = "score (e.g. a "
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s    
    elif "low score (" in text:
        p = "low score ("
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s    
    elif "the score is " in text:
        p = "the score is "
        text = text[text.index(p)+len(p):].strip()
        s = float(text[:2])
        score = s         
    else:
        # other.append(text)
        # if "grade" in text.lower(): other_score.append(text)
        pass

# print(len(other), len(other_score))
# for x in other_score[1:]:
#     print("\n", x)
#     break

    if score < S_MIN:
        return S_MIN
    elif score > S_MAX:
        return S_MAX
    else:
        return score

df["pred"] = df["llm"].apply(llm_to_score)

print(df["pred"].describe().to_dict())

df_test = df.copy()["essay_id pred".split()]

df_test.to_csv(f"data/test.csv")
