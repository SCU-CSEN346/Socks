import pickle
import numpy as np
from string import punctuation
import re
import time
import pandas as pd
import subprocess

# !pip install -U spacy
subprocess.call(['pip', 'install', "-U", "spacy"])
import spacy

# !python -m spacy download es_core_news_md
subprocess.call(['python', '-m', "spacy", "download", "es_core_news_md"])
nlp = spacy.load('es_core_news_md')

# !pip install unidecode
subprocess.call(['pip', 'install', "unidecode"])
from unidecode import unidecode

import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

# !pip install Levenshtein
subprocess.call(['pip', 'install', "Levenshtein"])
import Levenshtein as lev

from class_numbercheckspeller import SymSpellNumbers
base_symspell = pickle.load(open("base_symspell.pickle", "rb"))
D_correction = pickle.load(open("D_correction.pickle", "rb"))
ws = SymSpellNumbers(base_symspell, D_correction)

s = "tresiento pedrosequedo con trespartes lose por quesume"# hora 
ws.find(s)

dic_rae_ud = pickle.load(open("resource_rae_ud.pickle", "rb"))
keywords_C1 = ["miss", "tio", "mami", "mamita", "papa", "papi", "papito", "tia", "nose", "no se", "mama", "profe", "profesora", "profesor", "no ce", "noce", "hola", "ola", "chao", "me"]
slang_C1 = ["momo", "wtf", "kk", "lol", "cul", "miau", "po", "f", "chat", "hello", "bye", "okey", "ok", "hi", "paco", "caca", "pene", "mames"]
generated_faces = pickle.load(open("faces.pickle", "rb"))

punct = punctuation+"´"+"¡"
vowel = "aeiou"
digit = "0123456789"
blank = " "
math_punct = """$%()*+,-./:<=>[\]{}x"""
list_rae = dic_rae_ud["rae"]
list_ud = dic_rae_ud["ud"]

from nltk.corpus import stopwords
spanish_stopwords = stopwords.words("spanish")

punct_faces = []
for f in generated_faces:
    if all(c in punct for c in f):
        punct_faces.append(f)
        
no_digit_faces = []
for f in generated_faces:
    if not f.isdigit():
        no_digit_faces.append(f)
        
def replace_multiple(string, list_replace, replace_ch):
    for ch in list_replace:
        if ch in string:
            string = string.replace(ch, replace_ch)
    return string

def get_relevants_subjects(q):
    oo = {"PROPN": [], "NOUN": []}
    bi_oo = {"NOUN-ADP": [], "NOUN-PROPN": []}
    nlp_q = []
    for t in nlp(q):
        r = t.text, t.tag_, t.dep_, t.is_alpha, t.is_stop
        nlp_q.append(r)
    for r in nlp_q:
        if r[1] == "PROPN":
            oo["PROPN"].append(r)
        elif r[1] == "NOUN":
            oo["NOUN"].append(r)
    for k in range(len(nlp_q)-1):
        r1, r2 = nlp_q[k], nlp_q[k+1]
        if r1[1] == "NOUN" and r2[1] == "ADP":
            bi_oo["NOUN-ADP"].append((r1, r2))
        elif r1[1] == "NOUN" and r2[1] == "PROPN":
            bi_oo["NOUN-PROPN"].append((r1, r2))
    relevants = []
    for x in oo["PROPN"]:
        if x[3]:
            relevants.append(x)
    for x in oo["NOUN"]:
        if x[2] in ["nsubj"]:
            relevants.append(x)
    for x, y in bi_oo["NOUN-PROPN"]:
        relevants.append(x)
        relevants.append(y)
    for x, y in bi_oo["NOUN-ADP"]:
        if x[2] not in ["nmod", "obj", "obl"]:
            relevants.append(x)
        if y[2] not in ["case"] and not y[4]:
            relevants.append(y)
    return list(set([x[0].lower() for x in relevants]))

def get_attrib(a):
    a = "" if a == "nan" else a
    rel_subj = get_relevants_subjects(a)
    a_org_tokens = a.split()
    a_lower_org_tokens = a.lower().split()
    a_lower_org_tokens_wo_punct = replace_multiple(" ".join(a_lower_org_tokens), punct, " ").split()
    a_ud = [x for x in a_org_tokens if (x.lower() in list_ud) and not (x.lower() in list_rae)]
    a = unidecode(a)
    a = a.replace("\n", " ")
    a = " ".join(a.lower().split())
    a = ws.apply(a)
    a = " ".join([x.strip() for x in re.split(r'(-?\d*\.?\d+)', a)])
    lf_a = [(t.text, t.lemma_, t.tag_) for t in nlp(" ".join(a_org_tokens))]
    for propn_ in ["matemáticas", "matemáticas", "matemática", "matematica", "matematicas","sofia", "renata", "camila", "pia","pía", "carla", "pamela", "patricia", "matilde", "nama", "mamá", "amigo", "amiga", "amigos", "amigas"]:
        if propn_ in a_lower_org_tokens+a_org_tokens+a.split():
            lf_a.append((propn_, propn_, "PROPN"))
    aux_list = []
    if any([x in " ".join(a_lower_org_tokens) for x in ["lo mismo", "la misma", "los dos", "las dos", "los 2", "las 2"]]) or any([x in a for x in  ["lo mismo", "la misma", "los dos", "las dos", "los 2", "las 2"]]):
        aux_list.append("ambos")
        
    return {
        "clean": a,
        "org_tokens": a_org_tokens,
        "lower_org_tokens": a_lower_org_tokens,
        "org_tokens_wo_punct": a_lower_org_tokens_wo_punct, 
        "tokens": a.split(),
        "blank": list(map(a.lower().count, blank))[0],
        "vowel": list(map(a.lower().count, vowel)),
        "punct": list(map(a.lower().count, punct)),
        "math_punct": list(map(a.lower().count, math_punct)),
        "digit": list(map(a.lower().count, digit)),
        "numbers": re.findall(r"\d+", a),
        "no_numbers": [t for t in a.split() if str(t).isalpha()],
        "ud": a_ud, 
        "rae": [x for x in a.split() if x in list_rae],
        "faces": [f for f in generated_faces if f in "".join(a_org_tokens)],
        "slang": [f for f in slang_C1 if f in "".join(a_org_tokens)],
        "keywords": [f for f in keywords_C1 if f in "".join(a_org_tokens)],
        "lf_propn": list({lf_w[0].lower() for lf_w in lf_a if lf_w[-1] == "PROPN"}),
        "lf_lemma": list({lf_w[1].lower() for lf_w in lf_a}),
        "rel_subj": list(set(rel_subj+[lf_w[0] for lf_w in lf_a if lf_w[-1] == "PROPN"])),
        "aux_tokens": aux_list
    }

def get_simple_topo(dic_a):
    a = dic_a["clean"]
    tokens = dic_a["tokens"]
    blank = dic_a["tokens"]
    numbers = dic_a["numbers"]
    digit = dic_a["digit"]
    o =  {
        "len": len(a),
        "num_tokens": len(tokens),
        "num_numbers": len(numbers),
        "num_math_punct": sum(dic_a["math_punct"]),
        "num_digit": sum(digit),
        "num_rae": len(dic_a["rae"]),
        "num_ud": len(dic_a["ud"]),
        "num_punct": sum(dic_a["punct"]),
        "num_slang": len(dic_a["slang"]),
        "num_faces": len(dic_a["faces"]),
        "num_keywords": len(dic_a["keywords"]),
        "num_no_numbers": len(dic_a["no_numbers"])
    }
    return o

def get_ratio_vowel(dic_a):
    a = dic_a["clean"]
    count_vowel = dic_a["vowel"]
    count_blank = dic_a["blank"]
    count_punct = dic_a["punct"]
    count_digit = dic_a["digit"]
    if sum(count_vowel) > 0:
        return sum(count_vowel) / (len(a)-count_blank-sum(count_digit)-sum(count_punct))
    else:
        return 0

def get_max_len_number(dic_a):
    if dic_a["numbers"]:
        return max([len(n) for n in dic_a["numbers"]])
    else:
        return 0

def get_ratio_punct(dic_a, default=True):
    a = dic_a["clean"]
    count_blank = dic_a["blank"]
    count_punct = dic_a["punct"] if default else dic_a["math_punct"]
    if sum(count_punct) > 0:
        return sum(count_punct) / (len(a)-count_blank)
    else:
        return 0

def get_ratio_rae_ud(dic_a, l):
    if l=="rae" and len(dic_a["org_tokens"])>0:
        return len(dic_a[l])/len(dic_a["org_tokens"])
    elif l=="ud" and len(dic_a["tokens"])>0:
        return len(dic_a[l])/len(dic_a["tokens"])
    else:
        return 0

def get_ratio_faces(dic_a):
    if len(dic_a["org_tokens"])>0:
        return len(dic_a["faces"]) / len(dic_a["org_tokens"])
    else:
        return 0

def get_ratio_slang(dic_a):
    if len(dic_a["org_tokens"])>0:
        return len(dic_a["slang"]) / len(dic_a["org_tokens"])
    else:
        return 0

def get_ratio_keywords(dic_a):
    if len(dic_a["org_tokens"])>0:
        return len(dic_a["keywords"]) / len(dic_a["org_tokens"])
    else:
        return 0

def get_ratio_no_numbers(dic_a):
    if len(dic_a["org_tokens"])>0:
        return len(dic_a["no_numbers"]) / len(dic_a["org_tokens"])
    else:
        return 0 
    
def sim_lev(a, b):
    return 1 - lev.distance(a, b) / max(len(a), len(b)) if len(a) != 0 else 0

def get_injection_index(a, q):
    relevant_words = [w for w in replace_multiple(unidecode(q).lower(), punct, " ").split() if not w in spanish_stopwords]
    theta = 0.7
    tokens_a = a.split()
    injection_index = 0
    for k, t in enumerate(tokens_a):
        for s in relevant_words:
            if sim_lev(s, t) >= theta:
                injection_index += 1
                break
        if len(tokens_a):
            injection_index *= 1/len(relevant_words)
    return injection_index

def get_exist_numbs(dic_a, tresh=5):
    num_numbers = len(dic_a["numbers"])
    if num_numbers == 0:
        a = dic_a["clean"]
        if "poco" in a or "mucho" in a:
            return 2
        else:
            return int(any([(d in a) for d in "1234567890"]))
    else:
        return int(max(len(str(n)) for n in dic_a["numbers"])<tresh)
    
def get_sim_keywords(dic_a, keywords, theta=0.7):
    a = dic_a["clean"]
    tokens_a = a.split()
    injection_index = 0
    for k, t in enumerate(tokens_a):
        for s in keywords:
            if sim_lev(s, t) >= theta:
                injection_index += 1
                break
    return injection_index

def get_sim_implication(dic_a, dic_q, typo="3"):
    # esta correcto lo que dijo/dice? not-!quién este en él correcto > (si, no)
    # cual de las dos afirmaciones esta correcta? (alguna de las afirmaciones)
    # quien esta en lo correcto? (quién estar en él correcto) > any PROPN
    theta = 0.7
    q_lemma = [w.replace(",", "").replace(".", "") for w in dic_q["lf_lemma"]]
    injection_index = 0
    if typo=="3":
        if (("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and "correcto" in q_lemma) or ("tener" in q_lemma and "razón" in q_lemma):
            if ("quién" in q_lemma) or ("cuál" in q_lemma):
                propn_q = dic_q["lf_propn"]+["ninguno","ninguna", "todos","todas", "ambos","ambas", "nadie", "alguno", "alguna"]
                tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["aux_tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
                for k, t in enumerate(tokens_a):
                    t = t.replace(",", "").replace(".", "")
                    for s in propn_q:
                        if sim_lev(s, t) >= theta:
                            injection_index += 1
                            break
            else:
                tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"]+dic_a["org_tokens_wo_punct"])
                for k, t in enumerate(tokens_a):
                    t = t.replace(",", "").replace(".", "")
                    for s in ["falso", "verdadero", "sip", "nop", "estamal", "estabien", "bien", "mal", "si", "no", "correcta", "confundida", "confundido", "correcto", "equivocada", "equivocado", "incorrecta", "incorrecto", "razon", "razón"]:
                        if sim_lev(s, t) >= theta:
                            injection_index += 1
                            break
        elif ("por" in q_lemma and "qué" in q_lemma and ("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and ("equivocado" in q_lemma or "equivocada" in q_lemma)):
            tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in ["porque", "por", "que", "bien", "sip", "nop", "estamal", "estabien", "mal", "si", "no", "correcta", "confundida", "confundido", "correcto", "equivocada", "equivocado", "incorrecta", "incorrecto", "razon", "razón"]:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
        elif ("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and ("bien" in q_lemma):
            tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in ["verdadero", "falso", "bien", "mal", "si", "no", "correcta", "correcto", "equivocada", "equivocado", "confundida", "confundido", "incorrecta", "incorrecto", "razon", "razón", "sip", "nop", "estamal", "estabien"]:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
    elif typo=="4":
        if "ser" in q_lemma and "posible" in q_lemma:
            tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in ["verdadero", "falso", "bien", "mal", "si", "no", "correcta", "correcto", "equivocada", "equivocado", "confundida", "confundido", "incorrecta", "incorrecto", "razon", "razón", "sip", "nop", "estamal", "estabien"]:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
            
        elif "quién" in q_lemma or "cuál" in q_lemma or "qué" in q_lemma:
            propn_q = dic_q["rel_subj"]+["ninguno","ninguna", "todos","todas", "ambos","ambas", "nadie", "alguno", "alguna"]
            tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["aux_tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in propn_q:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
    return injection_index
            
def get_overlap_by(dic_a, dic_q, by=""): #Q[propn+]&A, Q[quién|cuál|qué], Q[ser&posible], A[binary(si|no)]
    theta = 0.7
    injection_index = 0
    if "Q" in by and "A" not in by:
        q_lemma = [w.replace(",", "").replace(".", "") for w in dic_q["lf_lemma"]]
        if by == "Q[quién|cuál|qué]":
            if "quién" in q_lemma or "cuál" in q_lemma or "qué" in q_lemma:
                injection_index = 1
        elif by == "Q[ser&posible]":
            if "ser" in q_lemma and "posible" in q_lemma:
                injection_index = 1
        elif by == "Q[(ser*&correcto)|(tener&razón)]":
            if (("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and "correcto" in q_lemma) or ("tener" in q_lemma and "razón" in q_lemma):
                injection_index = 1
        elif by == "Q[ser*&correcto]":
            if ("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and "correcto" in q_lemma:
                injection_index = 1
        elif  by == "Q[tener&razón]":
             if "tener" in q_lemma and "razón" in q_lemma:
                injection_index = 1
        elif by == "Q[quién|cuál]":
            if "quién" in q_lemma or "cuál":
                injection_index = 1
        elif by == "Q[ser*&bien]":
            if ("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and "bien" in q_lemma:
                injection_index = 1
        elif by == "Q[por&qué&ser*&equivocado]":
            if "por" in q_lemma and "qué" in q_lemma and ("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and ("equivocado" in q_lemma or "equivocada" in q_lemma):
                injection_index = 1
        elif by == "Q[por&qué&equivocado]":
            if "por" in q_lemma and "qué" in q_lemma  and ("equivocado" in q_lemma or "equivocada" in q_lemma):
                injection_index = 1
        
        
    elif "A" in by and "Q" not in by:
        tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
        if by == "A[binary(si|no)]":
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in ["verdadero", "falso", "bien", "mal", "si", "no", "correcta", "correcto", "equivocada", "equivocado", "incorrecta", "confundida", "confundido", "incorrecto", "razon", "razón", "sip", "nop", "estamal", "estabien"]:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
        if by == "A[porque|binary(si, no)]":
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in ["porque", "por", "que", "bien", "sip", "nop", "estamal", "estabien", "mal", "si", "no", "correcta", "correcto", "confundida", "confundido", "equivocada", "equivocado", "incorrecta", "incorrecto", "razon", "razón"]:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
    elif "A" in by and "Q" in by:
        if by == "Q[rel_subj]+&A+":
            propn_q = dic_q["rel_subj"]+["ninguno","ninguna", "todos","todas", "ambos","ambas", "nadie", "alguno", "alguna"]
            tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["aux_tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in propn_q:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
        if by == "Q[propn]+&A+":
            propn_q = dic_q["rel_subj"]+["ninguno","ninguna", "todos","todas", "ambos","ambas", "nadie", "alguno", "alguna"]
            tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["aux_tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in propn_q:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
    return injection_index

def get_topo_features(a):
    def get_crit_prop_vowel(w, prop_vowels):
        if len(w.split()) == 1:
            l_ = len(w.replace(" ", ""))
            if l_ and prop_vowels > 0.675:
                return 1
            elif w.isalpha():
                l_prop = {5: 2/5, 6: 2/6, 7: 2/7, 8: 3/8, 9: 4/9, 10: 4/10, 11: 5/11, 12: 6/12, 13: 5/13, 14: 6/14, 15: 8/15, 16: 8/16, 17: 8/17, 18: 9/18, 19: 9/19, 20: 9/20}
                if l_>21 and prop_vowels<1/2: 
                    return 1
                elif 21>l_>4:
                    return int(prop_vowels<l_prop[l_])
        return 0

    def get_prop_vowels(w):
        N = len(a.replace(" ", ""))
        if N>0:
            return sum( int(w in "aeiou") for w in a.replace(" ", "")) / N
        else:
            return 0
    
    def len_max_rep_char(w):
        w=w+" "
        c0 = w[0]
        lens = [0]
        clen = 1
        for c in w[1:]:
            if c == c0:
                clen += 1
            else:
                if c0.isalpha():
                    if clen>3 and c0 in ["r", "l", "c"]:
                        lens.append(clen)
                    elif clen>1:
                        lens.append(clen)
                c0 = c
                clen = 1
        return max(lens)  
    
    def max_char_fre_per_token(w, c="k"):
        tw = w.split()
        fmax = 0
        for t in tw:
            f = sum(int(ch==c) for ch in t)
            if f>fmax:
                fmax = f
        return fmax
    
    def max_type_rep_char_per_token(w, t="vowel"):
        w=unidecode(w+" ")
        c0 = w[0]
        lens = [0]
        clen = 1
        for c in w[1:]:
            if (c0.isalpha() and c.isalpha()) and ((c in "aeiou" and c0 in "aeiou") or (c not in "aeiou" and c0 not in "aeiou")):
                clen += 1
            else:
                if t=="vowel":
                    if c0 in "aeiou":
                        lens.append(clen)
                else:
                    if c0 not in "aeiou":
                        lens.append(clen) 
                c0 = c
                clen = 1
        return max(lens) 
    
    a = str(a).replace("\n", " ").lower()
    a = " ".join(a.split())
    o = {}
    
    na = a.replace(" ", "")
    
    o["len(~A)"] = len(na)
    o["prop_punct"] = sum(int(w in punct) for w in na)/o["len(~A)"] if o["len(~A)"]>0 else 0
    o["prop_punct+no-vowel"] = sum(int(w in punct or (w not in "aeiou" and w.isalpha())) for w in na)/o["len(~A)"] if o["len(~A)"]>0 else 0
    o["prop_vowels"] = get_prop_vowels(a)
    o["len(tokens(A))"] = len(a.split())
    o["len_max_rep_char"] = len_max_rep_char(a)
    o["A.isface()"] = int(a in generated_faces and not a.isdigit() and a not in ["ANA", "ana"])
    o["A.isdigit()"] = int(na.isdigit())
    o["frec_char(k)"] = max_char_fre_per_token(a, c="k")
    o["frec_char(g)"] = max_char_fre_per_token(a, c="g")
    o["frec_char(y)"] = max_char_fre_per_token(a, c="y")
    o["frec_char(j)"] = max_char_fre_per_token(a, c="j")
    o["frec_char(h)"] = max_char_fre_per_token(a, c="h")
    o["frec_char(x)"] = max_char_fre_per_token(a, c="x")
    o["frec_char(w)"] = max_char_fre_per_token(a, c="w")
    o["frec_char(ñ)"] = max_char_fre_per_token(a, c="ñ")
    o["A.is(nose)"] = int(a == "nose")
    o["A.is(nan)"] = int(a == "nan")
    o["A.is(ola|hola)"] = int(a in ["hola", "ola"])
    o["A.contains(bad-word)"] = sum(int(w in ["chupalo", "chupala", "puta", "puto", "caca", "pene", "kk"]) for w in a.split())
    o["A.contains(punct_faces)"] = sum(int(f in a) for f in punct_faces)
    o["prop_punct+digit"] = sum(int(w in punct or w.isdigit())for w in na)/o["len(~A)"] if o["len(~A)"]>0 else 0
    o["prop_no_math_punct"] = sum(int(w not in math_punct and w in punct )for w in na)/o["len(~A)"] if o["len(~A)"]>0 else 0
    o["max_no-vowel_rep_char_per_token"] = max_type_rep_char_per_token(a, "")
    o["prop_no_digit_faces"] = sum(int(t in no_digit_faces and t not in ["ANA", "ana"]) for t in a.split())/len(a.split())
    o["prop_keywords"] = sum(int(t in ["n0c3", "n0s3", "H3LL0", "paco", "momo", "love", "noce", "oki", "nosee", "okey", "help", "chao", "nose", "lel", "lol", "hola"]) for t in a.split())/len(a.split())
    o["prop_digit_char"] = sum(int(t.isdigit()) for t in na)/o["len(~A)"]
    o["max_vowel_rep_char_per_token"] = max_type_rep_char_per_token(a, "vowel")
    o["prop_no_digit_no_math_punct"] = sum(int(c in math_punct and not c.isdigit()) for c in a.replace(" ", ""))/o["len(~A)"]
    o["num_alpha"] = sum(int(t.isalpha()) for t in a.replace(" ", ""))
    o["prop_alpha_vowels"] = sum(int(t in "aeiou") for t in a.replace(" ", ""))/o["num_alpha"] if o["num_alpha"]>0 else 0
    
    return o

def get_overlap_by(dic_a, dic_q, by=""): #Q[propn+]&A, Q[quién|cuál|qué], Q[ser&posible], A[binary(si|no)]
    theta = 0.7
    injection_index = 0
    if "Q" in by and "A" not in by:
        q_lemma = [w.replace(",", "").replace(".", "") for w in dic_q["lf_lemma"]]
        if by == "Q[quién|cuál|qué]":
            if "quién" in q_lemma or "cuál" in q_lemma or "qué" in q_lemma:
                injection_index = 1
        elif by == "Q[ser&posible]":
            if "ser" in q_lemma and "posible" in q_lemma:
                injection_index = 1
        elif by == "Q[(ser*&correcto)|(tener&razón)]":
            if (("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and "correcto" in q_lemma) or ("tener" in q_lemma and "razón" in q_lemma):
                injection_index = 1
        elif by == "Q[ser*&correcto]":
            if ("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and "correcto" in q_lemma:
                injection_index = 1
        elif  by == "Q[tener&razón]":
             if "tener" in q_lemma and "razón" in q_lemma:
                injection_index = 1
        elif by == "Q[quién|cuál]":
            if "quién" in q_lemma or "cuál":
                injection_index = 1
        elif by == "Q[ser*&bien]":
            if ("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and "bien" in q_lemma:
                injection_index = 1
        elif by == "Q[por&qué&ser*&equivocado]":
            if "por" in q_lemma and "qué" in q_lemma and ("ser" in q_lemma or "es" in q_lemma or "estar" in q_lemma or "este" in q_lemma or "esta" in q_lemma) and ("equivocado" in q_lemma or "equivocada" in q_lemma):
                injection_index = 1
        elif by == "Q[por&qué&equivocado]":
            if "por" in q_lemma and "qué" in q_lemma  and ("equivocado" in q_lemma or "equivocada" in q_lemma):
                injection_index = 1
        
        
    elif "A" in by and "Q" not in by:
        tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
        if by == "A[binary(si|no)]":
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in ["verdadero", "falso", "bien", "mal", "si", "no", "correcta", "correcto", "equivocada", "equivocado", "incorrecta", "confundida", "confundido", "incorrecto", "razon", "razón", "sip", "nop", "estamal", "estabien"]:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
        if by == "A[porque|binary(si, no)]":
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in ["porque", "por", "que", "bien", "sip", "nop", "estamal", "estabien", "mal", "si", "no", "correcta", "correcto", "confundida", "confundido", "equivocada", "equivocado", "incorrecta", "incorrecto", "razon", "razón"]:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
    elif "A" in by and "Q" in by:
        if by == "Q[rel_subj]+&A+":
            propn_q = dic_q["rel_subj"]+["ninguno","ninguna", "todos","todas", "ambos","ambas", "nadie", "alguno", "alguna"]
            tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["aux_tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in propn_q:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
        if by == "Q[propn]+&A+":
            propn_q = dic_q["rel_subj"]+["ninguno","ninguna", "todos","todas", "ambos","ambas", "nadie", "alguno", "alguna"]
            tokens_a = set(dic_a["org_tokens"]+dic_a["tokens"]+dic_a["aux_tokens"]+dic_a["lower_org_tokens"]+dic_a["org_tokens_wo_punct"])
            for k, t in enumerate(tokens_a):
                t = t.replace(",", "").replace(".", "")
                for s in propn_q:
                    if sim_lev(s, t) >= theta:
                        injection_index += 1
                        break
    return injection_index

class Preprocessing(object):
    def __init__(self):
        self.dic_comparison = {}
        self.dic_lf = {}
        self.dic_topo = {}
        self.dic_attrib = {"a": {}, "q": {}}
        self.dic_features = {}
        self.dic_simple_topo = {}
    
    def get_dic_attrib(self, t, ix, a):
        if ix not in self.dic_attrib[t].keys():    
            self.dic_attrib[t][ix] = get_attrib(a)
        return self.dic_attrib[t][ix]
            
    def get_lf(self, ix, a):
        a = "" if a == "nan" else a
        a = a.replace("\n", " ")
        a = " ".join(a.split())
        if ix not in self.dic_lf.keys():
            o = {}
            doc_a = nlp(a)   
            num_tokens = len(doc_a)
            o["num_tokens"] = num_tokens
            for token in doc_a:
                dic_lf = {
#                     "lemma": token.lemma_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "shape": token.shape_,
                    "is_alpha": token.is_alpha,
                    "is_stop": token.is_stop
                }
                for k, v in dic_lf.items():
                    name_col = f"{k}<&>{v}"
                    if name_col not in o.keys():
                        o[name_col] = 0
                    o[name_col] += 1
                    
                    if k == "shape":
                        if v != "" and unidecode(v) == "":
                            name_col = f"{k}<&>emoji"
                            if name_col not in o.keys():
                                o[name_col] = 0
                            o[name_col] += 1 
                        else:
                            for c in set(v):
                                name_col = f"{k}<&>contains({c})"
                                if c in """!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~""":
                                    name_col = f"{k}<&>contains(punct)"
                                
                                if name_col not in o.keys():
                                    o[name_col] = 0
                                o[name_col] += 1

            for k, v in o.copy().items():
                if (k != "num_tokens") and ("lemma<&>" not in k) and (("shape<&>" not in k) or ("shape<&>" in k and k.split("<&>")[1] in ["emoji", "contains(x)", "contains(d)", "contains(punct)"])):
                    o[f"ratio({k}/num_tokens)"] = v/o["num_tokens"] if o["num_tokens"]!= 0 else 0
                    
            self.dic_lf[ix] = o
        return self.dic_lf[ix]
        
    def get_representation(self, ixa, ixq, a, q):
        ix = f"{ixa}_{ixq}" 
        if ix not in self.dic_comparison.keys():
            dic_a = self.get_dic_attrib("a", ixa, a)  
            dic_q = self.get_dic_attrib("q", ixq, q)  
            o = {}
            o["Q[rel_subj]+&A+"] = get_overlap_by(dic_a, dic_q, "Q[rel_subj]+&A+")
#             o["Q[quién|cuál|qué]"] = get_overlap_by(None, dic_q, "Q[quién|cuál|qué]")
#             o["Q[ser&posible]"] = get_overlap_by(None, dic_q, "Q[ser&posible]")
            o["A[binary(si|no)]"] = get_overlap_by(dic_a, None, "A[binary(si|no)]")

#             o["Q[por&qué&equivocado]"] = get_overlap_by(None, dic_q, "Q[por&qué&equivocado]")
#             o["Q[ser*&bien]"] = get_overlap_by(None, dic_q, "Q[ser*&bien]")
            o["A[porque|binary(si|no)]"] = get_overlap_by(dic_a, None, "A[porque|binary(si|no)]")
            o["Q[propn]+&A+"] = get_overlap_by(dic_a, dic_q, "Q[propn]+&A+")
#             o["Q[quién|cuál]"] = get_overlap_by(None, dic_q, "Q[quién|cuál]")
#             o["Q[(ser*&correcto)|(tener&razón)]"] = get_overlap_by(None, dic_q, "Q[(ser*&correcto)|(tener&razón)]")
#             o["Q[ser*&correcto]"] = get_overlap_by(None, dic_q, "Q[ser*&correcto]")
#             o["Q[tener&razón]"]= get_overlap_by(None, dic_q, "Q[tener&razón]")

            o["injection_index"] = get_injection_index(str(a), str(q))
            self.dic_comparison[ix] = o
        return self.dic_comparison[ix] 
            
    def get_preprocessing(self, D):
        data_representations = []
        _times = []
        for k, ixa in enumerate(D.index):
            start = time.time()
            a = D.loc[ixa]["respuesta"]
            a = str(a)
            q = D.loc[ixa]["pregunta"]
            q = str(q) 
            ixq = D.loc[ixa]["pregunta_id"]
            o = self.get_features(ixa, ixq, a, q)
            data_representations.append(o)
            end = time.time()
            _times.append(end-start)
            time_expected = (len(D.index)-(k+1))*np.mean(_times)
            time_expected_min = np.floor(time_expected/60)
            time_expected_sec = time_expected - time_expected_min*60
            print(f"""{k+1}/{len(D.index)}, progress: {100*(k+1)/len(D.index): .2f} %, dt: {_times[-1]: .2f}, exp. dt: {np.mean(_times): .2f} p/m {np.std(_times): .2f} s, t. trans: {np.sum(_times)/60: .1f} min, t. exp. end: {time_expected_min: .1f} m {time_expected_sec: .1f} s""")            
        df_representation = pd.DataFrame(data_representations, index=D.index)
        fillna_0 = ["lemma<&>", "tag<&>", "dep<&>", "shape<&>", "is_alpha<&>", "is_stop<&>"]
        dic_fillna = {c: 0 for c in df_representation.columns if any(x in c for x in fillna_0)}
        df_representation = df_representation.fillna(dic_fillna)
        return df_representation
    
    def get_topo(self, ix, a):
        if ix not in self.dic_topo.keys():
            self.dic_topo[ix] = get_topo_features(a)
        return self.dic_topo[ix]   
    
    def get_simple_topo(self, ix, a):
        if ix not in self.dic_simple_topo.keys():
            dic_a = self.get_dic_attrib("a", ix, a)   
            self.dic_simple_topo[ix] = get_simple_topo(dic_a)
        return self.dic_simple_topo[ix]   
        
    def get_features(self, ixa, ixq, a, q):
            ix = f"{ixa}_{ixq}"
            if ix not in self.dic_features.keys():
                o = self.get_representation(ixa, ixq, a, q)
                o = {**o, **self.get_topo(ixa, a)}
                o = {**o, **self.get_lf(ixa, a)}
                dic_a = self.get_dic_attrib("a", ixa, a)   
                for k, v in self.get_simple_topo(ixa, a).items():
                    if k in [
                             'len',
                             'num_digit',
                             'num_faces',
                             'num_keywords',
                             'num_math_punct',
                             'num_no_numbers',
                             'num_numbers',
                             'num_punct',
                             'num_rae',
                             'num_slang',
                             'num_tokens',
                             'num_ud'
                    ]: o[k] = v
                    
                o["ratio_rae"] = get_ratio_rae_ud(dic_a, "rae")
                o["ratio_ud"] = get_ratio_rae_ud(dic_a, "ud")
                o["ratio_vowel"] = get_ratio_vowel(dic_a)
                o["ratio_no_numbers"] = get_ratio_no_numbers(dic_a)
                o["ratio_punct"] =  get_ratio_punct(dic_a, default=True)
                    
                o["exist_numbs"] =  get_exist_numbs(dic_a)
                o["ratio_slang"] = get_ratio_slang(dic_a)
                o["ratio_keywords"] = get_ratio_keywords(dic_a)
                o["ratio_faces"] = get_ratio_faces(dic_a)
                o["max_len_number"] = get_max_len_number(dic_a)
                
                self.dic_features[ix] = o
            return self.dic_features[ix]
        
 