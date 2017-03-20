import numpy as np
from pandas.core.config import option_context
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.classification import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append('/home/juan/git/PUCV-projects/textos/src')
from vector_representation import build_tree_classifier, load_data


import sys
import optparse



class AClassifier:
    def __init__(self, inner_classifier_initialization):
        self.clf = inner_classifier_initialization

    def init_overall_variables(self):
        self.run_aucmeans = []
        self.negclass_f1 = []
        self.negclass_precision = []
        self.negclass_recall = []
        self.posclass_f1 = []
        self.posclass_precision = []
        self.posclass_recall = []

        self.eval_negclass_f1 = []
        self.eval_negclass_precision = []
        self.eval_negclass_recall = []
        self.eval_posclass_f1 = []
        self.eval_posclass_precision = []
        self.eval_posclass_recall = []

    def init_run_variables(self):
        self.mean_tpr = 0.0
        self.mean_fpr = np.linspace(0, 1, 100)
        self.negclass_f1_sum = 0.0
        self.negclass_precision_sum = 0.0
        self.negclass_recall_sum = 0.0
        self.posclass_f1_sum = 0.0
        self.posclass_precision_sum = 0.0
        self.posclass_recall_sum = 0.0

        self.eval_negclass_f1_sum = 0.0
        self.eval_negclass_precision_sum = 0.0
        self.eval_negclass_recall_sum = 0.0
        self.eval_posclass_f1_sum = 0.0
        self.eval_posclass_precision_sum = 0.0
        self.eval_posclass_recall_sum = 0.0




parser = optparse.OptionParser()
parser.add_option('--tr', '--training-path', dest='training', help='Folder path were the training data is located')
parser.add_option('--ts', '--testing-path', dest='testing', help='Folder path were the testing data is located')
parser.add_option('-o', '--output-path', dest='output', help='Output path were the ROC figures will be saved')
parser.add_option('-r', '--nruns', dest='nruns', help='Number of runs to perform (training data is shuffled and a k-fold cv is performed in each one)')
parser.add_option('-k', '--kfold', dest='kfold', help='Number of folds that will be created in each CV process.')


(options, args) = parser.parse_args()

if options.training is None or options.testing is None or options.output is None or options.nruns is None or options.kfold is None:
    print("Usage:"+sys.argv[0]+" --tr training_data_path --ts testing_data_path -o output_path -r nruns -k kfolds")
    sys.exit(-1)



# Parameters
output_path = options.output
training_data_path = options.training
testing_data_path = options.testing
nruns = int(options.nruns)
k = int(options.kfold)
#output_path = '/home/juan/git/PUCV-projects/textos/data/testing'
# training_data_path = '/home/juan/git/PUCV-projects/textos/data/training'
#testing_data_path = '/home/juan/git/PUCV-projects/textos/data/testing'
#nruns = 5


# Loading training and testing data
X_tr,y_tr,features_train = load_data(training_data_path)
X_test, y_test, features_test  = load_data(testing_data_path)


classifiers = dict()
#classifiers["DecisionTreeClassifier(criterion='entropy', max_depth=5)"] = AClassifier(tree.DecisionTreeClassifier(criterion='entropy', max_depth=5))

#classifiers["KNeighborsClassifier(3)"] = AClassifier(KNeighborsClassifier(3))

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]


classifiers["GridSearchCV(estimator=SVC(kernel='rbf', probability=True), param_grid=param_grid, cv=3)"] = AClassifier(
    GridSearchCV(estimator=SVC(probability=True), param_grid=param_grid, cv=3, scoring='f1'))

##classifiers["SVC(kernel='rbf', probability=True, C=1)"] = AClassifier(SVC(kernel='rbf', probability=True))
##classifiers["SVC(kernel='rbf', probability=True, gamma=2, C=1)"] = AClassifier(SVC(gamma=2, probability=True, C=1))
##classifiers["SVC(kernel='linear', probability=True, C=0.025)"] = AClassifier(SVC(kernel="linear", probability=True, C=0.025))

#classifiers["RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)"] = AClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
#classifiers["AdaBoostClassifier()"] = AClassifier(AdaBoostClassifier())

#classifiers["GaussianNB()"] = AClassifier(GaussianNB())
#classifiers["MultinomialNB()"] = AClassifier(MultinomialNB())



'''
run_aucmeans = []
negclass_f1 = []
negclass_precision = []
negclass_recall = []
posclass_f1 = []
posclass_precision = []
posclass_recall = []

eval_negclass_f1 = []
eval_negclass_precision = []
eval_negclass_recall = []
eval_posclass_f1 = []
eval_posclass_precision = []
eval_posclass_recall = []
'''


for name_c, c in classifiers.items():
    c.init_overall_variables()


for run in range(1, nruns+1):
    # Shuffling the training data for each run
    X_train, y_train = shuffle(X_tr, y_tr)

    # Balancing the dataset (training and testing)
    sm = SMOTE(random_state=42)
    X_tt, y_tt = sm.fit_sample(X_train, y_train)
    X_ts, y_ts = sm.fit_sample(X_test, y_test)

    #print('Original  dataset shape {}'.format(Counter(y_train)))
    #print('Resampled dataset shape {}'.format(Counter(y_tt)))

    #clf = build_tree_classifier(X_tt, y_tt, features_train, target_names = ['schizo', 'non-schizo'], path='/Volumes/SSDII/Users/juan/git/PUCV-projects/textos/data/training')

    assert(X_tt.shape[1] == X_ts.shape[1] ) # if it fails then the features file does not match between train and test data
    #probas_ =clf.predict_proba(X_test)

    n_samples, n_features = X_tt.shape
    random_state = np.random.RandomState(0)
    cv = StratifiedKFold(n_splits=k)

    ##clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)
    ##clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

    '''
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    '''

    y_tt = np.array(map(int,y_tt))
    y_ts = np.array(map(int,y_ts))
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2


    i = 0

    '''
    negclass_f1_sum = 0.0
    negclass_precision_sum = 0.0
    negclass_recall_sum = 0.0
    posclass_f1_sum = 0.0
    posclass_precision_sum = 0.0
    posclass_recall_sum = 0.0

    eval_negclass_f1_sum = 0.0
    eval_negclass_precision_sum = 0.0
    eval_negclass_recall_sum = 0.0
    eval_posclass_f1_sum = 0.0
    eval_posclass_precision_sum = 0.0
    eval_posclass_recall_sum = 0.0
    '''
    for name_c, c in classifiers.items():
        c.init_run_variables()


    for (train, test), color in zip(cv.split(X_tt, y_tt), colors):
        #predicted_ = clf.fit(X_tt[train], y_tt[train]).predict(X_tt[test])
        #print(metrics.classification_report(y[test], predicted_) )

        for name_c, c in classifiers.items():
            model = c.clf.fit(X_tt[train], y_tt[train])
            print "\n",c.clf.get_params(),"\n"
            probas_ = model.predict_proba(X_tt[test])
            y_test_split = model.predict(X_tt[test])

            # precision, recall, F-measure and support
            precision, recall, f1, support = precision_recall_fscore_support(y_tt[test], y_test_split)

            c.negclass_f1_sum += f1[0]
            c.negclass_precision_sum += precision[0]
            c.negclass_recall_sum += recall[0]

            c.posclass_f1_sum += f1[1]
            c.posclass_precision_sum += precision[1]
            c.posclass_recall_sum += recall[1]

            # Evaluating over the test set
            y_test_split = model.predict(X_ts)

            # precision, recall, F-measure and support
            precision, recall, f1, support = precision_recall_fscore_support(y_ts, y_test_split)

            c.eval_negclass_f1_sum += f1[0]
            c.eval_negclass_precision_sum += precision[0]
            c.eval_negclass_recall_sum += recall[0]

            c.eval_posclass_f1_sum += f1[1]
            c.eval_posclass_precision_sum += precision[1]
            c.eval_posclass_recall_sum += recall[1]

            #print "Precision:", precision, "Recall:", recall, "F1:", f1, "Support:", support
            #print(metrics.classification_report(y_tt[test], y_test_split, target_names=['Schizo', 'Non-Schizo']) )

            fpr, tpr, thresholds = roc_curve(y_tt[test], probas_[:, 1])
            c.mean_tpr += interp(c.mean_fpr, fpr, tpr)
            c.mean_tpr[0] = 0.0
            #roc_auc = auc(fpr, tpr)
        #plt.plot(fpr, tpr, lw=lw, color=color,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1

    for name_c, c in classifiers.items():
        #plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Luck')
        c.mean_tpr /= cv.get_n_splits(X_tt, y_tt)
        c.mean_tpr[-1] = 1.0
        c.mean_auc = auc(c.mean_fpr, c.mean_tpr)

        #plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

        c.run_aucmeans.append(c.mean_auc)

        c.negclass_f1.append(c.negclass_f1_sum/i)
        c.negclass_precision.append(c.negclass_precision_sum/i)
        c.negclass_recall.append(c.negclass_recall_sum/i)
        c.posclass_f1.append(c.posclass_f1_sum/i)
        c.posclass_precision.append(c.posclass_precision_sum/i)
        c.posclass_recall.append(c.posclass_recall_sum/i)

        c.eval_negclass_f1.append(c.eval_negclass_f1_sum/i)
        c.eval_negclass_precision.append(c.eval_negclass_precision_sum/i)
        c.eval_negclass_recall.append(c.eval_negclass_recall_sum/i)
        c.eval_posclass_f1.append(c.eval_posclass_f1_sum/i)
        c.eval_posclass_precision.append(c.eval_posclass_precision_sum/i)
        c.eval_posclass_recall.append(c.eval_posclass_recall_sum/i)


    '''
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (run %d)'% run)
    plt.legend(loc="lower right")
    plt.savefig(output_path+'/'+'roc_schizo_r'+str(run)+'.png')
    #plt.show()
    plt.close()
    '''

    #predicted_ = clf.fit(X_tt, y_tt).predict(X_ts)
    #print(metrics.classification_report(y_ts, predicted_, target_names=['Schizo','Non-Schizo']) )

print "* %d runs were executed and within each run a %d-Fold CV was performed." % (nruns, k)
print "* At the end of each CV step an evaluation set (testing data) was presented to the classifier. Average performance attained is presented."
print "* Within each run the training data was shuffled."
print "* Average performance measures computed over all run means (each run generated a k-fold average)\n"

for name_c, c in classifiers.items():
    run_aucmeans = np.array(c.run_aucmeans)
    negclass_f1 = np.array(c.negclass_f1)
    negclass_precision = np.array(c.negclass_precision)
    negclass_recall = np.array(c.negclass_recall)
    posclass_f1 = np.array(c.posclass_f1)
    posclass_precision = np.array(c.posclass_precision)
    posclass_recall = np.array(c.posclass_recall)

    eval_negclass_f1 = np.array(c.eval_negclass_f1)
    eval_negclass_precision = np.array(c.eval_negclass_precision)
    eval_negclass_recall = np.array(c.eval_negclass_recall)
    eval_posclass_f1 = np.array(c.eval_posclass_f1)
    eval_posclass_precision = np.array(c.eval_posclass_precision)
    eval_posclass_recall = np.array(c.eval_posclass_recall)

    print ""
    print "Classifier: ", name_c
    if name_c.startswith('Grid'):
        best_parameters = c.clf.best_estimator_.get_params()
        for param_name in sorted(best_parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print "Performance over the training set:"
    print "Mean AUC (computed over all runs) is %0.4f(%0.4f)" % (np.mean(run_aucmeans), np.std(run_aucmeans))
    print '{:<10} {:<12} {:<12} {:<12}'.format('class','F1','Precision','Recall')
    print '{:<10} {:<12} {:<12} {:<12}'.format('----------','------------','------------','------------')
    print '{:<10} {:.3f}({:.3f}) {:.3f}({:.3f}) {:.3f}({:.3f})'.format('Schizo', np.mean(negclass_f1), np.std(negclass_f1),
                                         np.mean(negclass_precision), np.std(negclass_precision),
                                         np.mean(negclass_recall), np.std(negclass_recall))
    print '{:<10} {:.3f}({:.3f}) {:.3f}({:.3f}) {:.3f}({:.3f})'.format('Non-schizo', np.mean(posclass_f1), np.std(posclass_f1),
                                         np.mean(posclass_precision), np.std(posclass_precision),
                                         np.mean(posclass_recall), np.std(posclass_recall))
    print ""
    print "Performance over the evaluation set:"
    print '{:<10} {:<12} {:<12} {:<12}'.format('class','F1','Precision','Recall')
    print '{:<10} {:<12} {:<12} {:<12}'.format('----------','------------','------------','------------')
    print '{:<10} {:.3f}({:.3f}) {:.3f}({:.3f}) {:.3f}({:.3f})'.format('Schizo', np.mean(eval_negclass_f1), np.std(eval_negclass_f1),
                                         np.mean(eval_negclass_precision), np.std(eval_negclass_precision),
                                         np.mean(eval_negclass_recall), np.std(eval_negclass_recall))
    print '{:<10} {:.3f}({:.3f}) {:.3f}({:.3f}) {:.3f}({:.3f})'.format('Non-schizo', np.mean(eval_posclass_f1), np.std(eval_posclass_f1),
                                         np.mean(eval_posclass_precision), np.std(eval_posclass_precision),
                                         np.mean(eval_posclass_recall), np.std(eval_posclass_recall))
