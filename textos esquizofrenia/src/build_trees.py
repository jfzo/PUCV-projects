from sklearn import tree
import sys
import numpy as np
sys.path.append('/home/juan/git/PUCV-projects/textos/src')
from vector_representation import build_tree_classifier, load_data
from imblearn.over_sampling import SMOTE
import pydotplus



training_data_path1 = '/home/juan/git/PUCV-projects/textos/data/Relato_C'
training_data_path2 = '/home/juan/git/PUCV-projects/textos/data/Relato_B'
training_data_path3 = '/home/juan/git/PUCV-projects/textos/data/Relato_A'

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

'''
testing_data_path1 = '/home/juan/git/PUCV-projects/textos/data/ab/'
testing_data_path2 = '/home/juan/git/PUCV-projects/textos/data/ac/'
testing_data_path3 = '/home/juan/git/PUCV-projects/textos/data/bc/'
X_ts1,y_ts1,_ = load_data(testing_data_path1)
X_ts2,y_ts2,_ = load_data(testing_data_path2)
X_ts3,y_ts3,_ = load_data(testing_data_path3)
'''



clf1 = tree.DecisionTreeClassifier(class_weight= None, criterion= 'gini', max_depth= 10 ,max_features= None, max_leaf_nodes= None, min_impurity_split= 1e-07, min_samples_leaf= 1, min_samples_split= 6, min_weight_fraction_leaf= 0.0, presort= False, random_state= None, splitter= 'best')
clf1 = clf1.fit(X_ttr1,y_ttr1)
dot_data1 = tree.export_graphviz(clf1, out_file=None,feature_names=features_train1,class_names=['non-schizo','schizo'],filled=True, rounded=True,special_characters=True)
graph1 = pydotplus.graph_from_dot_data(dot_data1)
graph1.write_pdf("dataset1.pdf")


clf2 = tree.DecisionTreeClassifier(class_weight= None, criterion= 'entropy', max_depth= 5, max_features= None, max_leaf_nodes= None, min_impurity_split= 1e-07, min_samples_leaf= 5, min_samples_split= 2, min_weight_fraction_leaf= 0.0, presort= False, random_state= None, splitter= 'best')
clf2 = clf2.fit(X_ttr2,y_ttr2)
dot_data2 = tree.export_graphviz(clf2, out_file=None,feature_names=features_train1,class_names=['non-schizo','schizo'],filled=True, rounded=True,special_characters=True)
graph2 = pydotplus.graph_from_dot_data(dot_data2)
graph2.write_pdf("dataset2.pdf")


clf3 = tree.DecisionTreeClassifier(class_weight= None, criterion= 'entropy', max_depth= 10, max_features= None, max_leaf_nodes= None, min_impurity_split= 1e-07, min_samples_leaf= 1, min_samples_split= 4, min_weight_fraction_leaf= 0.0, presort= False, random_state= None, splitter= 'best')
clf3 = clf3.fit(X_ttr3,y_ttr3)
dot_data3 = tree.export_graphviz(clf3, out_file=None,feature_names=features_train1,class_names=['non-schizo','schizo'],filled=True, rounded=True,special_characters=True)
graph3 = pydotplus.graph_from_dot_data(dot_data3)
graph3.write_pdf("dataset3.pdf")





