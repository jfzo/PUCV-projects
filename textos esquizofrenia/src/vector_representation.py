def load_data(path="."):
    import numpy as np

    X = np.loadtxt(path+"/"+"doc_vectors.csv", delimiter=",")
    y = [l.strip().split(" ")[1] for l in open(path+"/"+'classes')]
    fp = open(path+"/"+'features')
    features = [l.strip() for l in fp]
    fp.close()
    return X, y, features


def build_classifier(X, y, features, path="."):
    import numpy as np
    from sklearn import tree
    import pydotplus
    from collections import Counter
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X, y)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
    clf = clf.fit(X_res, y_res)
    target_names = np.array(['schizo', 'non-schizo'])#target classes in ascending numerical order
    #target_names[np.where(y_res == '-1')] = 'schizo'
    #target_names[np.where(y_res == '1')] = 'nonschizo'
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=features,
                                    class_names=target_names,
                                    filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(path+"/"+"schizo.pdf")

    return X_res, y_res, target_names, clf

def vectorize(path='.'):
    import numpy as np
    import json

    # Load the inverted index with the features and the freqs:
    with open(path+'/'+'inv_index.json', 'r') as fp:
        inv_ix = json.load(fp)
    # Load the max tag frequency in each file
    with open(path+'/'+'max_tagfreq.json', 'r') as fp:
        max_tagfreq= json.load(fp)


    doc_ids= [l.strip().split(" ")[0] for l in open(path+'/'+'classes')]
    #doc_ids = max_tagfreq.keys()

    Ndocs = len(max_tagfreq)
    Nfeatures = len(inv_ix)
    feature_ids = list(inv_ix.keys())
    feature_ids.sort()

    # Filter features if it's necessary
    '''
    To select meta-tags (Nouns, Determiner)
    '''
    selected_features = [u'N']
    # META-TAGS
    # [u'A', u'C', u'D', u'F', u'I', u'N', u'P'
    # , u'R', u'S', u'V', u'W', u'Z']
    #[u'adjective', u'conjunction', u'determiner', u'punctuation', u'interjection', u'noun', u'pronoun',
    # u'adverb', u'adposition', u'verb', u'date', u'number']
    buffer_feat = []
    for i in range(len(feature_ids)):
        if feature_ids[i][0] in selected_features:
            buffer_feat.append(feature_ids[i])

    feature_ids = buffer_feat
    Nfeatures = len(feature_ids)

    # Saving features in the employed order
    with open(path+'/'+'features','w') as fp:
        for t in feature_ids:
            fp.write(t+'\n')

    print "Using features:",'/'.join(selected_features)
    print "Features saved in file 'features'"

    X = np.zeros((Ndocs,Nfeatures))

    for i in range(Ndocs):
        docid_i = doc_ids[i] # gets the name of the document
        for feat in feature_ids:
            idf_feat = np.log(float(Ndocs) / len(inv_ix[feat]))
            tf_i = inv_ix[feat].get(docid_i, 0.0) / float(max_tagfreq[docid_i])

            j = feature_ids.index(feat)
            X[i, j] = tf_i * idf_feat


    np.savetxt(path+'/'+"doc_vectors.csv", X, delimiter=",",fmt='%.6e')


def vectorize_with_previous_features(path='.'):
    print "vectorize_with_previous_features()"
    import numpy as np
    import json

    # Load the inverted index with the features and the freqs:
    with open(path+'/'+'inv_index.json', 'r') as fp:
        inv_ix = json.load(fp)
    # Load the max tag frequency in each file
    with open(path+'/'+'max_tagfreq.json', 'r') as fp:
        max_tagfreq= json.load(fp)

    tagsf = open(path+'/'+'features')
    features = [l.strip() for l in tagsf]
    tagsf.close()
    features.sort()
    Nfeatures = len(features)

    doc_ids= [l.strip().split(" ")[0] for l in open(path+'/'+'classes')]
    #doc_ids = max_tagfreq.keys()

    Ndocs = len(max_tagfreq)

    #feature_ids = list(inv_ix.keys())
    #feature_ids.sort()


    X = np.zeros((Ndocs,Nfeatures))

    for i in range(Ndocs):
        docid_i = doc_ids[i] # gets the name of the document
        for feat in features:
            if feat in inv_ix:
                idf_feat = np.log(float(Ndocs) / len(inv_ix[feat]))
                tf_i = inv_ix[feat].get(docid_i, 0.0) / float(max_tagfreq[docid_i])

                j = features.index(feat)
                X[i, j] = tf_i * idf_feat

    np.savetxt(path+'/'+"doc_vectors.csv", X, delimiter=",",fmt='%.6e')

if __name__ == "__main__":
    import pickle
    import sys
    #main()
    use_existing_features = False

    if len(sys.argv) > 1 and sys.argv[1] == 'use-features':
        use_existing_features = True
        print "Ensure that features file was already copied testing folder"


    if use_existing_features:
        vectorize_with_previous_features(path='/Volumes/SSDII/Users/juan/git/PUCV-projects/textos/data/testing')
    else:
        vectorize(path='/Volumes/SSDII/Users/juan/git/PUCV-projects/textos/data/training')
    #X,y,features = load_data()
    #X_res, y_res, target_names, clf = build_classifier(X,y,features)
    #pickle.dump((X_res, y_res, target_names, clf), open("model.p","w"))
