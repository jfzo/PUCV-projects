# load the tags file with the features
# visit each 'pairs' file and generate the vector.

import numpy as np
import json
import matplotlib.pyplot as plt

features = [l.strip() for l in open('tags')]

# Load the inverted index with the features and the freqs:
with open('inv_index.json', 'r') as fp:
    inv_ix = json.load(fp)
# Load the max tag frequency in each file
with open('max_tagfreq.json', 'r') as fp:
    max_tagfreq= json.load(fp)


doc_ids= [l.strip().split(" ")[0] for l in open('classes')]
#doc_ids = max_tagfreq.keys()

Ndocs = len(max_tagfreq)

S = np.zeros((Ndocs,Ndocs))

for i in range(Ndocs - 1):
    docid_i = doc_ids[i]
    S[i, i] = 1.0
    for j in range(i+1, Ndocs):
        docid_j = doc_ids[j]
        for feat in inv_ix:
            idf_feat = np.log(float(Ndocs) / len(inv_ix[feat]))
            tf_i = inv_ix[feat].get(docid_i,0)/float(max_tagfreq[docid_i])
            tf_j = inv_ix[feat].get(docid_j,0)/float(max_tagfreq[docid_j])
            S[i,j] += tf_i*tf_j*idf_feat
            S[j, i] = S[i,j]


np.savetxt("doc_similarities.csv", S, delimiter=",",fmt='%.6e')
print "Matrix saved to 'doc_similarities.csv'"
print "plotting..."
plt.matshow(S, cmap=plt.cm.gray)
plt.show()