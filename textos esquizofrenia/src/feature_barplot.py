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

    fset11 = set(np.array(features_train1)[np.argsort(clf1.feature_log_prob_[0, :])][-K:])
    #fset12 = set(np.array(features_train1)[np.argsort(clf1.feature_log_prob_[1, :])][-K:])

    fset21 = set(np.array(features_train2)[np.argsort(clf2.feature_log_prob_[0, :])][-K:])
    #fset22 = set(np.array(features_train2)[np.argsort(clf2.feature_log_prob_[1, :])][-K:])

    fset31 = set(np.array(features_train3)[np.argsort(clf3.feature_log_prob_[0, :])][-K:])
    #fset32 = set(np.array(features_train3)[np.argsort(clf3.feature_log_prob_[1, :])][-K:])

    #resulting_features = fset11 & fset12 & fset21 & fset22 & fset31 & fset32

    resulting_features = fset11 & fset21 & fset31

    return ",".join(resulting_features)


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

    print select_pos_features(int(sys.argv[1]))


















