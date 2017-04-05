# load the tags file with the features
# visit each 'pairs' file and generate the vector.

def complete_pos_processing():
    from os import listdir
    import json

    print "*******************************\nUSING ALL POS FEATURES\n*******************************\n"

    tagsf = open('tags')
    features = [l.strip() for l in tagsf]
    tagsf.close()



    ctrlset = [f for f in listdir('Control') if f.endswith('pairs')]
    xprmset = [f for f in listdir('Experimental') if f.endswith('pairs')]

    # It can also be initialized from a file.
    inv_ix = dict() # tags as keys and a list of document names as values.
    max_tagf = dict()

    for ctrl_i in ctrlset:
        posfile = open('Control/'+ctrl_i)
        original_fname = ctrl_i[:-6]
        max_tagf_i = 0 # to store the maximum frequency of a tag (useful to normalize the weights)
        # Each line has this format: 'A:B\n', where 'A' denotes a term and 'B' denotes a pos tag
        tknstags = [tuple(l.strip().split(":")) for l in posfile][1:] #first one is removed 'cause it is associated to the number of document
        posfile.close()
        #Do something to filter by tag or word if it's necessary!
        for _, tag in tknstags:
            if not tag in inv_ix:
                inv_ix[tag] = dict()
            inv_ix[tag][original_fname] = inv_ix[tag].setdefault(original_fname, 0) + 1 # increases the freq of the tag in the file.
            if inv_ix[tag][original_fname] > max_tagf_i:
                max_tagf_i = inv_ix[tag][original_fname]

        max_tagf[original_fname] = max_tagf_i

    for xprm_i in xprmset:
        posfile = open('Experimental/'+xprm_i)
        original_fname = xprm_i[:-6]
        max_tagf_i = 0  # to store the maximum frequency of a tag (useful to normalize the weights)
        # Each line has this format: 'A:B\n', where 'A' denotes a term and 'B' denotes a pos tag
        tknstags = [tuple(l.strip().split(":")) for l in posfile][1:] #first one is removed 'cause it is associated to the number of document
        posfile.close()
        #Do something to filter by tag or word if it's necessary!
        for _, tag in tknstags:
            if not tag in inv_ix:
                inv_ix[tag] = dict()
            inv_ix[tag][original_fname] = inv_ix[tag].get(original_fname, 0) + 1 # increases the freq of the tag in the file.
            if inv_ix[tag][original_fname] > max_tagf_i:
                max_tagf_i = inv_ix[tag][original_fname]

        max_tagf[original_fname] = max_tagf_i


    with open('inv_index.json', 'w') as fp:
        json.dump(inv_ix, fp)

    with open('max_tagfreq.json', 'w') as fp:
        json.dump(max_tagf, fp)





def meta_pos_processing():
    '''
    Instead of just using the complete feature set, this function employs only the meta tags, i.e. the initials of
    the tags. For instance, instead of using VAG0000 and VSIS3S0 as features, both will contribute to the feature V,
    that represents the verbs.

    '''
    from os import listdir
    import json

    print "*******************************\nUSING META POS FEATURES\n*******************************\n"

    tagsf = open('tags')
    features = [l.strip() for l in tagsf]
    tagsf.close()


    ctrlset = [f for f in listdir('Control') if f.endswith('pairs')]
    xprmset = [f for f in listdir('Experimental') if f.endswith('pairs')]

    # It can also be initialized from a file.
    inv_ix = dict() # tags as keys and a list of document names as values.
    max_tagf = dict()

    for ctrl_i in ctrlset:
        posfile = open('Control/'+ctrl_i)
        original_fname = ctrl_i[:-6]
        max_tagf_i = 0 # to store the maximum frequency of a tag (useful to normalize the weights)
        # Each line has this format: 'A:B\n', where 'A' denotes a term and 'B' denotes a pos tag
        tknstags = [tuple(l.strip().split(":")) for l in posfile][1:] #first one is removed 'cause it is associated to the number of document
        posfile.close()
        #Do something to filter by tag or word if it's necessary!
        for _, tag in tknstags:
            tag = tag[0] ################################ ONLY FIRST CHAR IS NECESSARY
            if not tag in inv_ix:
                inv_ix[tag] = dict()
            inv_ix[tag][original_fname] = inv_ix[tag].setdefault(original_fname, 0) + 1 # increases the freq of the tag in the file.
            if inv_ix[tag][original_fname] > max_tagf_i:
                max_tagf_i = inv_ix[tag][original_fname]

        max_tagf[original_fname] = max_tagf_i


    for xprm_i in xprmset:
        posfile = open('Experimental/'+xprm_i)
        original_fname = xprm_i[:-6]
        max_tagf_i = 0  # to store the maximum frequency of a tag (useful to normalize the weights)
        # Each line has this format: 'A:B\n', where 'A' denotes a term and 'B' denotes a pos tag
        tknstags = [tuple(l.strip().split(":")) for l in posfile][1:] #first one is removed 'cause it is associated to the number of document
        posfile.close()
        #Do something to filter by tag or word if it's necessary!
        for _, tag in tknstags:
            tag = tag[0] ################################ ONLY FIRST CHAR IS NECESSARY
            if not tag in inv_ix:
                inv_ix[tag] = dict()
            inv_ix[tag][original_fname] = inv_ix[tag].get(original_fname, 0) + 1 # increases the freq of the tag in the file.
            if inv_ix[tag][original_fname] > max_tagf_i:
                max_tagf_i = inv_ix[tag][original_fname]

        max_tagf[original_fname] = max_tagf_i



    with open('inv_index.json', 'w') as fp:
        json.dump(inv_ix, fp)

    with open('max_tagfreq.json', 'w') as fp:
        json.dump(max_tagf, fp)





'''
# To load the dictionary from the file:
with open('inv_index.json', 'r') as fp:
    inv_ix = json.load(fp)
'''

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'use-meta-features':
        print "Using meta POS tags as features."
        meta_pos_processing()
    elif len(sys.argv) > 1 and sys.argv[1] == 'use-all-features':
        print "Using all POS tags as features."
        complete_pos_processing()
    else:
        print "Usage:",sys.argv[0],"use-meta-features|use-all-features"
        sys.exit()


