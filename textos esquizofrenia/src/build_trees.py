from sklearn import tree
import sys
import numpy as np
sys.path.append('/home/juan/git/PUCV-projects/textos/src')
from vector_representation import build_tree_classifier, load_data
from imblearn.over_sampling import SMOTE
import pydotplus



training_data_path1 = '/home/juan/git/PUCV-projects/textos/data/ab'
testing_data_path1 = '/home/juan/git/PUCV-projects/textos/data/Relato_C/'

training_data_path2 = '/home/juan/git/PUCV-projects/textos/data/ac'
testing_data_path2 = '/home/juan/git/PUCV-projects/textos/data/Relato_B/'

training_data_path3 = '/home/juan/git/PUCV-projects/textos/data/bc'
testing_data_path3 = '/home/juan/git/PUCV-projects/textos/data/Relato_A/'

X_tr1,y_tr1,features_train1 = load_data(training_data_path1)
X_ts1,y_ts1,_ = load_data(testing_data_path1)
y_tr1 = np.array(map(int,y_tr1))
y_ts1 = np.array(map(int,y_ts1))
# generating a big dataset with training and testing samples
X1 = np.concatenate((X_tr1, X_ts1))
Y1 = np.concatenate((y_tr1, y_ts1))

X_tr2,y_tr2,features_train2 = load_data(training_data_path2)
X_ts2,y_ts2,_ = load_data(testing_data_path2)
y_tr2 = np.array(map(int,y_tr2))
y_ts2 = np.array(map(int,y_ts2))
# generating a big dataset with training and testing samples
X2 = np.concatenate((X_tr2, X_ts2))
Y2 = np.concatenate((y_tr2, y_ts2))

X_tr3,y_tr3,features_train3 = load_data(training_data_path3)
X_ts3,y_ts3,_ = load_data(testing_data_path3)
y_tr3 = np.array(map(int,y_tr3))
y_ts3 = np.array(map(int,y_ts3))
# generating a big dataset with training and testing samples
X3 = np.concatenate((X_tr3, X_ts3))
Y3 = np.concatenate((y_tr3, y_ts3))


sm = SMOTE(random_state=42)
'''
# Resampling only training data
X1,Y1 = sm.fit_sample(X_tr1, y_tr1)
X2,Y2 = sm.fit_sample(X_tr2, y_tr2)
X3,Y3 = sm.fit_sample(X_tr3, y_tr3)
'''
# Resampling the entire datasets
X1,Y1 = sm.fit_sample(X1, Y1)
X2,Y2 = sm.fit_sample(X2, Y2)
X3,Y3 = sm.fit_sample(X3, Y3)





clf1 = tree.DecisionTreeClassifier(class_weight= None,criterion= 'entropy',max_depth= 4,max_features= None,max_leaf_nodes= None,min_impurity_split= 1e-07,min_samples_leaf= 9,min_samples_split= 6,min_weight_fraction_leaf= 0.0,presort= False,random_state= 42,splitter= 'best')
clf1 = clf1.fit(X1,Y1)
dot_data1 = tree.export_graphviz(clf1, out_file=None,feature_names=features_train1,class_names=['ctrl','expr'],filled=True, rounded=True,special_characters=True)
graph1 = pydotplus.graph_from_dot_data(dot_data1)
graph1.write_pdf("tree-dataset-1.pdf")


clf2 = tree.DecisionTreeClassifier(class_weight= None,criterion= 'entropy',max_depth= 3,max_features= None,max_leaf_nodes= None,min_impurity_split= 1e-07,min_samples_leaf= 5,min_samples_split= 4,min_weight_fraction_leaf= 0.0,presort= False,random_state= 42,splitter= 'best')
clf2 = clf2.fit(X2,Y2)
dot_data2 = tree.export_graphviz(clf2, out_file=None,feature_names=features_train1,class_names=['ctrl','expr'],filled=True, rounded=True,special_characters=True)
graph2 = pydotplus.graph_from_dot_data(dot_data2)
graph2.write_pdf("tree-dataset-2.pdf")


clf3 = tree.DecisionTreeClassifier(class_weight= None,criterion= 'entropy',max_depth= 3,max_features= None,max_leaf_nodes= None,min_impurity_split= 1e-07,min_samples_leaf= 7,min_samples_split= 4,min_weight_fraction_leaf= 0.0,presort= False,random_state= 42,splitter= 'best')
clf3 = clf3.fit(X3,Y3)
dot_data3 = tree.export_graphviz(clf3, out_file=None,feature_names=features_train1,class_names=['ctrl','expr'],filled=True, rounded=True,special_characters=True)
graph3 = pydotplus.graph_from_dot_data(dot_data3)
graph3.write_pdf("tree-dataset-3.pdf")





