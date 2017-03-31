import sys
import numpy as np
sys.path.append('/home/juan/git/PUCV-projects/textos/src')
from vector_representation import build_tree_classifier, load_data
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB



def plot_bars(coefficients, features, name='example.pdf'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    freq_series = pd.Series.from_array(coefficients)

    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind='bar', fontsize=4, grid=True)
    ax.set_title("Coefficients of MultinomialNB")
    ax.set_xlabel("POS feature")
    ax.set_ylabel("coefficient")
    ax.set_xticklabels(features) 
    #plt.show()
    plt.savefig(name)


def generate_shared_barchart(dict_counts, fname):
    import numpy as np
    import matplotlib.pyplot as plt
    pos_names = dict(zip([u'A', u'C', u'D', u'F', u'I', u'N', u'P', u'R', u'S', u'V', u'W', u'Z'],
                         [u'adjective', u'conjunction', u'determiner', u'punctuation', u'interjection', u'noun',
                          u'pronoun', u'adverb', u'adposition', u'verb', u'date', u'number']))

    n_groups = len(dict_counts)
    ticks = sorted(dict_counts.keys())


    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    fig, ax = plt.subplots()
    rects1 = plt.bar(index, [dict_counts[t][0] for t in ticks], bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='Experimental')

    rects2 = plt.bar(index + bar_width, [dict_counts[t][1] for t in ticks], bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='Control')

    plt.xlabel('Meta Tag')
    plt.ylabel('Probability')
    plt.title('Probabilities of occurrence of tags given each text category')
    ticks = [pos_names[x] for x in ticks]
    plt.xticks(index + bar_width / 2, ticks)

    plt.legend(loc='upper left')

    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=90, fontsize=10)


    plt.tight_layout()
    #plt.show()
    plt.savefig(fname)


def get_probabilistic_summary(sel_items, features, probabilities):
    #sel_items = np.argsort(clf1.feature_log_prob_[1, :])[-K:]
    l_feats_probs = sorted( [(np.array(features)[x], np.exp(probabilities[1, x])) for x in sel_items] )
    summary_counts = dict()
    for f,p in l_feats_probs:
        if f[0] not in summary_counts:
            summary_counts[f[0]] = [0.0, 1.0]
        summary_counts[f[0]][0] += p
        summary_counts[f[0]][1] *= p
    summary = dict()
    for f, counts in summary_counts.items():
        if counts[0] == counts[1]:
            counts[1] = 0.0
        else:
            summary[f] = counts[0] - counts[1]

    return summary



def select_pos_features(K = 50):
    training_data_path1 = '/home/juan/git/PUCV-projects/textos/data/ab'
    training_data_path2 = '/home/juan/git/PUCV-projects/textos/data/ac'
    training_data_path3 = '/home/juan/git/PUCV-projects/textos/data/bc'

    X_tr1, y_tr1, features_train1 = load_data(training_data_path1)
    X_tr2, y_tr2, features_train2 = load_data(training_data_path2)
    X_tr3, y_tr3, features_train3 = load_data(training_data_path3)
    y_tr1 = np.array(map(int, y_tr1))
    y_tr2 = np.array(map(int, y_tr2))
    y_tr3 = np.array(map(int, y_tr3))

    sm = SMOTE(random_state=42)
    X_ttr1, y_ttr1 = sm.fit_sample(X_tr1, y_tr1)
    X_ttr2, y_ttr2 = sm.fit_sample(X_tr2, y_tr2)
    X_ttr3, y_ttr3 = sm.fit_sample(X_tr3, y_tr3)

    clf1 = MultinomialNB(alpha=1.0)
    clf1 = clf1.fit(X_ttr1, y_ttr1)

    clf2 = MultinomialNB(alpha=1.0)
    clf2 = clf2.fit(X_ttr2, y_ttr2)

    clf3 = MultinomialNB(alpha=1.0)
    clf3 = clf3.fit(X_ttr3, y_ttr3)

    #                                     -- features sorted (ASC) according to dependence --
    fset11 = set( np.array(features_train1)[ np.argsort(clf1.feature_log_prob_[0, :]) ][-K:] )
    fset12 = set( np.array(features_train1)[ np.argsort(clf1.feature_log_prob_[1, :]) ][-K:] )
    D11 = get_probabilistic_summary(np.argsort(clf1.feature_log_prob_[0, :])[-K:],features_train1,clf1.feature_log_prob_)
    D12 = get_probabilistic_summary(np.argsort(clf1.feature_log_prob_[1, :])[-K:],features_train1,clf1.feature_log_prob_)

    bardict = dict()
    [bardict.setdefault(x, [D11.get(x,0.0), D12.get(x,0.0)]) for x in list(set(D11.keys()) | set(D12.keys()))]
    generate_shared_barchart(bardict,'/home/juan/git/PUCV-projects/textos/data/feature_prob_dataset-1.pdf')


    fset21 = set( np.array(features_train2)[ np.argsort(clf2.feature_log_prob_[0, :]) ][-K:] )
    fset22 = set( np.array(features_train2)[ np.argsort(clf2.feature_log_prob_[1, :]) ][-K:] )
    D21 = get_probabilistic_summary(np.argsort(clf2.feature_log_prob_[0, :])[-K:],features_train2,clf2.feature_log_prob_)
    D22 = get_probabilistic_summary(np.argsort(clf2.feature_log_prob_[1, :])[-K:],features_train2,clf2.feature_log_prob_)

    bardict = dict()
    [bardict.setdefault(x, [D21.get(x,0.0), D22.get(x,0.0)]) for x in list(set(D21.keys()) | set(D22.keys()))]
    generate_shared_barchart(bardict,'/home/juan/git/PUCV-projects/textos/data/feature_prob_dataset-2.pdf')

    fset31 = set( np.array(features_train3)[ np.argsort(clf3.feature_log_prob_[0, :]) ][-K:] )
    fset32 = set( np.array(features_train3)[ np.argsort(clf3.feature_log_prob_[1, :]) ][-K:] )
    D31 = get_probabilistic_summary(np.argsort(clf3.feature_log_prob_[0, :])[-K:],features_train3,clf3.feature_log_prob_)
    D32 = get_probabilistic_summary(np.argsort(clf3.feature_log_prob_[1, :])[-K:],features_train3,clf3.feature_log_prob_)

    bardict = dict()
    [bardict.setdefault(x, [D31.get(x,0.0), D32.get(x,0.0)]) for x in list(set(D31.keys()) | set(D32.keys()))]
    generate_shared_barchart(bardict,'/home/juan/git/PUCV-projects/textos/data/feature_prob_dataset-3.pdf')

    resulting_features = fset11 & fset12 & fset21 & fset22 & fset31 & fset32

    resulting_features_neg = fset11 & fset21 & fset31
    resulting_features_pos = fset12 & fset22 & fset32

    #####
    #print get_probabilistic_summary(np.argsort(clf1.feature_log_prob_[0, :])[-K:], features_train1, clf1.feature_log_prob_)

    return ",".join(sorted(resulting_features_neg)), ",".join(sorted(resulting_features_pos)),",".join(sorted(resulting_features)) # SELECTED FEATURES ASSOCIATED TO NEGATIVE, POSITIVE AND TO THE WHOLE DATA


def plot_bars_log_prob():
    training_data_path1 = '/home/juan/git/PUCV-projects/textos/data/ab'
    training_data_path2 = '/home/juan/git/PUCV-projects/textos/data/ac'
    training_data_path3 = '/home/juan/git/PUCV-projects/textos/data/bc'

    X_tr1,y_tr1,features_train1 = load_data(training_data_path1)
    X_tr2,y_tr2,features_train2 = load_data(training_data_path2)
    X_tr3,y_tr3,features_train3 = load_data(training_data_path3)
    y_tr1 = np.array(map(int,y_tr1))
    y_tr2 = np.array(map(int,y_tr2))
    y_tr3 = np.array(map(int,y_tr3))

    sm = SMOTE(random_state=42)
    X_ttr1,y_ttr1 = sm.fit_sample(X_tr1, y_tr1)
    X_ttr2,y_ttr2 = sm.fit_sample(X_tr2, y_tr2)
    X_ttr3,y_ttr3 = sm.fit_sample(X_tr3, y_tr3)



    clf1 = MultinomialNB(alpha=1.0)
    clf1 = clf1.fit(X_ttr1,y_ttr1)
    plot_bars(clf1.feature_log_prob_[0,:], features_train1, '/home/juan/git/PUCV-projects/textos/data/feature_coefficients_ds1_cl1.pdf')
    plot_bars(clf1.feature_log_prob_[1,:], features_train1, '/home/juan/git/PUCV-projects/textos/data/feature_coefficients_ds1_cl2.pdf')

    clf2 = MultinomialNB(alpha=1.0)
    clf2 = clf2.fit(X_ttr2,y_ttr2)
    plot_bars(clf2.feature_log_prob_[0,:], features_train2, '/home/juan/git/PUCV-projects/textos/data/feature_coefficients_ds2_cl1.pdf')
    plot_bars(clf2.feature_log_prob_[1,:], features_train2, '/home/juan/git/PUCV-projects/textos/data/feature_coefficients_ds2_cl2.pdf')


    clf3 = MultinomialNB(alpha=1.0)
    clf3 = clf3.fit(X_ttr3,y_ttr3)
    plot_bars(clf3.feature_log_prob_[0,:], features_train3, '/home/juan/git/PUCV-projects/textos/data/feature_coefficients_ds3_cl1.pdf')
    plot_bars(clf3.feature_log_prob_[1,:], features_train3, '/home/juan/git/PUCV-projects/textos/data/feature_coefficients_ds3_cl2.pdf')



if __name__ == "__main__":
    #plot_bars_log_prob()
    import sys

    sys.path.append('/home/juan/git/PUCV-projects/textos/src')
    from feature_barplot import select_pos_features

    print '\n\n'.join(select_pos_features(int(sys.argv[1])))


















