import numpy as np


def edit_distance(target, source, distances, c):
    m = len(target)+1
    n = len(source)+1
    D = [[0 for x in range(n)] for x in range(m)]
    for i in range(m):
        for j in range(n):
            if i == 0:
                D[i][j] = j * c
            elif j == 0:
                D[i][j] = i * c

            elif target[i-1] == source[j-1]:
                D[i][j] = D[i-1][j-1]
            else:
                D[i][j] = min(
                    c + D[i][j-1], 
                    c + D[i-1][j], 
                    distances[f"{target[i-1]}_{source[j-1]}"] + D[i-1][j-1]
                )    
    return D[m-1][n-1]


def similarity(target, source, distances, c, s, p):
    d = edit_distance(target, source, distances, c)
    eta = np.exp((-d / s) ** p)
    return eta


def predict_suffix(test_form, train_data, distances, c, s, p, tf=False):
    train_data_form = train_data[train_data.ipa_base!=test_form]

    ity_bases = train_data_form[train_data_form.suffix=="ity"].ipa_base.to_list()
    ness_bases = train_data_form[train_data_form.suffix=="ness"].ipa_base.to_list()

    ity_sims = [similarity(test_form, base, distances, c, s, p) for base in ity_bases]
    ness_sims = [similarity(test_form, base, distances, c, s, p) for base in ness_bases]

    if tf:
        ity_freqs = train_data_form[train_data_form.suffix=="ity"].frequency.astype(int).to_list()
        ness_freqs = train_data_form[train_data_form.suffix=="ness"].frequency.astype(int).to_list()
        assert len(ity_freqs) == len(ity_sims)
        assert len(ness_freqs) == len(ness_sims)
        ity_sims = [np.log10(f + 1) * s for f, s in zip(ity_freqs, ity_sims)]
        ness_sims = [np.log10(f + 1) * s for f, s in zip(ness_freqs, ness_sims)]

    score_ity = np.sum(ity_sims)
    score_ness = np.sum(ness_sims)
    if score_ity > score_ness:
        return "ity", score_ness - score_ity
    else:
        return "ness", score_ness - score_ity
