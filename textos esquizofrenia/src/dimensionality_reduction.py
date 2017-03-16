import numpy as np
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn import tree
import pydotplus
import string

features = [l.strip() for l in open('features')]
X = np.loadtxt("doc_vectors.csv", delimiter=",")
y = [l.strip().split(" ")[1] for l in open('classes')]

K = 20 # number of selected features

#kbestmod =  SelectKBest(chi2, k=K)
#kbestmod =  SelectKBest(f_classif, k=K)
kbestmod =  SelectKBest(mutual_info_classif, k=K)

model = kbestmod.fit(X,y)
selected_features = np.sort(np.argsort(model.pvalues_)[:K])
sel_feature_ids = [features[ft] for ft in selected_features]
# This is equivalent to: X_new = SelectKBest(chi2, k=10).fit_transform(X, y)
X_new = X[:, selected_features]
print "Reduced version of the data is stored in file 'reduced_doc_vectors.csv'"
np.savetxt("reduced_doc_vectors.csv", X_new, delimiter=",",fmt='%.6e')

with open('reduced_doc_vectors.features','w') as fp:
    for feat in sel_feature_ids:
        fp.write(feat+'\n')
print "Selected features saved in file 'reduced_doc_vectors.features'"


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_new, y)

doc_classes = string.replace(':'.join(y), '-1', 'ctrl').replace('1','expr').split(':')

tree.export_graphviz(clf, out_file='tree_fig.dot',
                         feature_names=sel_feature_ids,
                         class_names=doc_classes,
                         filled=True, rounded=True,
                         special_characters=True)
dot_data = ""
with open('tree_fig.dot') as fp:
    for l in fp:
        dot_data += l
os.remove('tree_fig.dot')

graph = pydotplus.graph_from_dot_data(dot_data)
print "Classification tree saved in file 'tree.pdf'"
graph.write_pdf("tree.pdf")
