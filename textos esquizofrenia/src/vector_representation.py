def load_data(path="."):
    '''
    Loads the data stored ad the doc_vectors.csv, classes files and the feature identifiers located at the given path
    :param path: Folder path
    :return: The data matrix, a vector with the corresponding labels and another vector containing the feature names.
    '''
    import numpy as np

    X = np.loadtxt(path+"/"+"doc_vectors.csv", delimiter=",")
    y = [l.strip().split(" ")[1] for l in open(path+"/"+'classes')]
    fp = open(path+"/"+'features')
    features = [l.strip() for l in fp]
    fp.close()
    return X, y, features


def build_tree_classifier(X, y, features,  target_names = ['negative', 'positive'], path=None):
    '''
    Builds a decision tree classifier and fits the data given. Optionally, it can draw the tree and store it as a PDF file
    :param X: Data matrix with one row per example and one column per feature
    :param y: Labels (one per example)
    :param features: Identifier of each feature
    :param target_names: Names of the labels in ascending order according to its number.
    :param path: If given, a pdf with the obtained tree is created at the specified folder path.
    :return: The trained classifier
    '''
    import numpy as np
    from sklearn import tree
    import pydotplus
    from collections import Counter
    from imblearn.over_sampling import SMOTE

    #sm = SMOTE(random_state=42)
    #X_res, y_res = sm.fit_sample(X, y)
    #print('Resampled dataset shape {}'.format(Counter(y_res)))
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
    clf = clf.fit(X, y)


    if path != None:
        target_names = np.array(target_names)  # target classes in ascending numerical order
        dot_data = tree.export_graphviz(clf, out_file=None,feature_names=features, class_names=target_names,filled=True, rounded=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(path+"/"+"schizo.pdf")

    return clf



def vectorize_with_specific_tags(selected_features, path='.'):
    import numpy as np
    import json

    # Load the inverted index with the features and the freqs:
    with open(path+'/'+'inv_index.json', 'r') as fp:
        inv_ix = json.load(fp)
    # Load the max tag frequency in each file
    with open(path+'/'+'max_tagfreq.json', 'r') as fp:
        max_tagfreq= json.load(fp)

    #print "The following features will be selected:",','.join(selected_features)
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
    #selected_features = [u'N']
    # META-TAGS
    # [u'A', u'C', u'D', u'F', u'I', u'N', u'P'
    # , u'R', u'S', u'V', u'W', u'Z']
    #[u'adjective', u'conjunction', u'determiner', u'punctuation', u'interjection', u'noun', u'pronoun',
    # u'adverb', u'adposition', u'verb', u'date', u'number']
    buffer_feat = []
    for i in range(len(feature_ids)):
        for sf in selected_features:
            if feature_ids[i].startswith(sf):
                buffer_feat.append(feature_ids[i])

    #for i in range(len(feature_ids)):
    #    if feature_ids[i][0] in selected_features:
    #        buffer_feat.append(feature_ids[i])

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

    if len(sys.argv) > 1:
        if sys.argv[1] == 'use-features':
            print "Using features collected in a previous processing."
            vectorize_with_previous_features(path='.')
        else:
            print "Using specific meta-pos-tags as features."
            vectorize_with_specific_tags(sys.argv[1].split(','), path='.')
    else:
        print "Usage:", sys.argv[0], "[use-features]|[comma_separated_list_of_pos_tags_or_initials]"
        sys.exit()


    #X,y,features = load_data()
    #X_res, y_res, target_names, clf = build_classifier(X,y,features)
    #pickle.dump((X_res, y_res, target_names, clf), open("model.p","w"))
