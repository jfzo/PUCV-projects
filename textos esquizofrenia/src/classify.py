import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

import sys
sys.path.append('/Volumes/SSDII/Users/juan/git/PUCV-projects/textos/src')
from vector_representation import build_classifier, load_data


X_train,y_train,features_train = load_data('/Volumes/SSDII/Users/juan/git/PUCV-projects/textos/data/training')

X_test,y_test,features_test = load_data('/Volumes/SSDII/Users/juan/git/PUCV-projects/textos/data/testing')


sm = SMOTE(random_state=42)
X_tt, y_tt = sm.fit_sample(X_train, y_train)
X_ts, y_ts = sm.fit_sample(X_test, y_test)

print('Original  dataset shape {}'.format(Counter(y_train)))
print('Resampled dataset shape {}'.format(Counter(y_tt)))

X_tt, y_tt, target_names, clf = build_classifier(X_tt, y_tt, features_train, '/Volumes/SSDII/Users/juan/git/PUCV-projects/textos/data/training')

assert(X_tt.shape[1] == X_ts.shape[1] ) # if it fails then the features file does not match between train and test data
#probas_ =clf.predict_proba(X_test)

'''
Loading data

X = np.loadtxt("doc_vectors.csv", delimiter=",")
y = [l.strip().split(" ")[1] for l in open('classes')]
'''

n_samples, n_features = X_tt.shape
random_state = np.random.RandomState(0)
cv = StratifiedKFold(n_splits=5)

#classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
#classifier = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)


y_tt = np.array(map(int,y_tt))
y_ts = np.array(map(int,y_ts))
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
for (train, test), color in zip(cv.split(X_tt, y_tt), colors):
    #predicted_ = clf.fit(X_tt[train], y_tt[train]).predict(X_tt[test])
    #print(metrics.classification_report(y[test], predicted_) )

    probas_ = clf.fit(X_tt[train], y_tt[train]).predict_proba(X_tt[test])

    fpr, tpr, thresholds = roc_curve(y_tt[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Luck')
mean_tpr /= cv.get_n_splits(X_tt, y_tt)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('/Volumes/SSDII/Users/juan/git/PUCV-projects/textos/data/training/roc_schizo.png')
#plt.show()


predicted_ = clf.fit(X_tt, y_tt).predict(X_ts)
print(metrics.classification_report(y_ts, predicted_) )

