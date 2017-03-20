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
import sys
sys.path.append('/home/juan/git/PUCV-projects/textos/src')
from vector_representation import build_tree_classifier, load_data


import sys
import optparse

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

    #clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)
    #clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)


    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)


    y_tt = np.array(map(int,y_tt))
    y_ts = np.array(map(int,y_ts))
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2


    i = 0

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

    for (train, test), color in zip(cv.split(X_tt, y_tt), colors):
        #predicted_ = clf.fit(X_tt[train], y_tt[train]).predict(X_tt[test])
        #print(metrics.classification_report(y[test], predicted_) )

        model = clf.fit(X_tt[train], y_tt[train])
        probas_ = model.predict_proba(X_tt[test])
        y_test_split = model.predict(X_tt[test])

        # precision, recall, F-measure and support
        precision, recall, f1, support = precision_recall_fscore_support(y_tt[test], y_test_split)

        negclass_f1_sum += f1[0]
        negclass_precision_sum += precision[0]
        negclass_recall_sum += recall[0]

        posclass_f1_sum += f1[1]
        posclass_precision_sum += precision[1]
        posclass_recall_sum += recall[1]

        # Evaluating over the test set
        y_test_split = model.predict(X_ts)

        # precision, recall, F-measure and support
        precision, recall, f1, support = precision_recall_fscore_support(y_ts, y_test_split)

        eval_negclass_f1_sum += f1[0]
        eval_negclass_precision_sum += precision[0]
        eval_negclass_recall_sum += recall[0]

        eval_posclass_f1_sum += f1[1]
        eval_posclass_precision_sum += precision[1]
        eval_posclass_recall_sum += recall[1]

        #print "Precision:", precision, "Recall:", recall, "F1:", f1, "Support:", support
        #print(metrics.classification_report(y_tt[test], y_test_split, target_names=['Schizo', 'Non-Schizo']) )

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

    run_aucmeans.append(mean_auc)

    negclass_f1.append(negclass_f1_sum/i)
    negclass_precision.append(negclass_precision_sum/i)
    negclass_recall.append(negclass_recall_sum/i)
    posclass_f1.append(posclass_f1_sum/i)
    posclass_precision.append(posclass_precision_sum/i)
    posclass_recall.append(posclass_recall_sum/i)

    eval_negclass_f1.append(eval_negclass_f1_sum/i)
    eval_negclass_precision.append(eval_negclass_precision_sum/i)
    eval_negclass_recall.append(eval_negclass_recall_sum/i)
    eval_posclass_f1.append(eval_posclass_f1_sum/i)
    eval_posclass_precision.append(eval_posclass_precision_sum/i)
    eval_posclass_recall.append(eval_posclass_recall_sum/i)



    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (run %d)'% run)
    plt.legend(loc="lower right")
    plt.savefig(output_path+'/'+'roc_schizo_r'+str(run)+'.png')
    #plt.show()
    plt.close()

    #predicted_ = clf.fit(X_tt, y_tt).predict(X_ts)
    #print(metrics.classification_report(y_ts, predicted_, target_names=['Schizo','Non-Schizo']) )

run_aucmeans = np.array(run_aucmeans)
negclass_f1 = np.array(negclass_f1)
negclass_precision = np.array(negclass_precision)
negclass_recall = np.array(negclass_recall)
posclass_f1 = np.array(posclass_f1)
posclass_precision = np.array(posclass_precision)
posclass_recall = np.array(posclass_recall)

eval_negclass_f1 = np.array(eval_negclass_f1)
eval_negclass_precision = np.array(eval_negclass_precision)
eval_negclass_recall = np.array(eval_negclass_recall)
eval_posclass_f1 = np.array(eval_posclass_f1)
eval_posclass_precision = np.array(eval_posclass_precision)
eval_posclass_recall = np.array(eval_posclass_recall)


print "* %d runs were executed and within each run a %d-Fold CV was performed." % (nruns, k)
print "* At the end of each CV step an evaluation set (testing data) was presented to the classifier. Average performance attained is presented."
print "* Within each run the training data was shuffled."
print "* Average performance measures computed over all run means (each run generated a k-fold average)\n"
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
