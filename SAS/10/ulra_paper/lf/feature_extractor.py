import pandas as pd
import re
import math
from textstat import textstat
import spacy
import time
import numpy as np

def calculate_surface_signals(text):
    ch_count = len(text)

    words = text.split()
    w_count = len(words)

    co_count = text.count(',')

    words = text.split()
    unique_words = set(words)
    uw_count = len(unique_words)

    return {
        "CH": ch_count,
        "W": w_count,
        "CO": co_count,
        "UW": uw_count
        }

def calculate_preposition_signals(doc):

    nn_count = 0
    dt_count = 0
    rb_count = 0
    jj_count = 0
    in_count = 0

    for token in doc:
        if token.pos_ == 'NOUN':
            nn_count += 1
        elif token.pos_ == 'DET':
            dt_count += 1
        elif token.pos_ == 'ADV':
            rb_count += 1
        elif token.pos_ == 'ADJ':
            jj_count += 1
        elif token.pos_ in ('ADP', 'SCONJ'):
            in_count += 1

    return {
        "NN": nn_count,
        "DT": dt_count,
        "RB": rb_count,
        "JJ": jj_count,
        "IN": in_count
    }

def calculate_readability_signals(text, doc, basic_vocab=[]):
    tokens = [token.text.lower() for token in doc]

    basic_vocab = set(basic_vocab)

    gf_index = textstat.gunning_fog(text)
    smog_index = textstat.smog_index(text)
    rix_index = textstat.rix(text)
    dc_index = textstat.dale_chall_readability_score(text)

    wt_count = len(set(tokens))

    s_count = textstat.sentence_count(text)

    lw_count = len([word for word in tokens if len(word) > 6])

    cw_count = len([word for word in tokens if textstat.syllable_count(word) > 2])

    nbw_count = len([word for word in tokens if word not in basic_vocab])

    dw_count = len([word for word in tokens if textstat.is_difficult_word(word)])

    return {
        "GF": gf_index,
        "SMOG": smog_index,
        "RIX": rix_index,
        "DC": dc_index,
        "WT": wt_count,
        "S": s_count,
        "LW": lw_count,
        "CW": cw_count,
        "NBW": nbw_count,
        "DW": dw_count
    }

class ULRAFeatureExtractor():

    def __init__(self, basic_vocab):
        self.nlp = spacy.load("en_core_web_sm")
        self.basic_vocab = basic_vocab

    def get_features(self, input_text):

        input_doc = self.nlp(input_text)

        surface_results = calculate_surface_signals(input_text)
        preposition_results = calculate_preposition_signals(input_doc)
        readability_results = calculate_readability_signals(input_text, input_doc, self.basic_vocab)

        o = {**surface_results, **preposition_results, **readability_results}

        return o

    def get_preprocessing(self, data, text_name="A"):

        raw_df = []
        _times = []

        for k, ixa in enumerate(data.index):
            start = time.time()
            a = data.loc[ixa][text_name]
            a = str(a)

            o = self.get_features(a)

            raw_df.append(o)

            end = time.time()
            _times.append(end-start)

            time_expected = (len(data.index)-(k+1))*np.mean(_times)
            time_expected_min = np.floor(time_expected/60)
            time_expected_sec = time_expected - time_expected_min*60

            if k % 100 == 0 or k==len(data.index)-1:
                print(f"""{k+1}/{len(data.index)}, progress: {100*(k+1)/len(data.index): .2f} %, dt: {_times[-1]: .2f}, exp. dt: {np.mean(_times): .2f} p/m {np.std(_times): .2f} s, t. trans: {np.sum(_times)/60: .1f} min, t. exp. end: {time_expected_min: .1f} m {time_expected_sec: .1f} s""")

        df = pd.DataFrame(raw_df, index=data.index)

        return df
