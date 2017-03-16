# count the number of words  (total and per category) in each document from the pos analysis made.
# Requires the .pairs files.

from os import listdir
import os
import json

tagsf = open('tags')
features = [l.strip() for l in tagsf]
tagsf.close()



ctrlset = [f for f in listdir('Control') if f.endswith('pairs')]
xprmset = [f for f in listdir('Experimental') if f.endswith('pairs')]
count_ix = {}

PATHFLDR = 'Control'

for doc in ctrlset+xprmset:
    if not os.path.exists(PATHFLDR+os.sep+doc):
        PATHFLDR = 'Experimental'
    fp = open(PATHFLDR+os.sep+doc)
    wcount = 0
    count_ix[doc[:-6]] = {}
    fp.readline() #to avoid counting the 1st line
    for l in fp:
        wcount += 1
        word, tag = l.strip().split(":")
        count_ix[doc[:-6]][tag[0]] = count_ix[doc[:-6]].get(tag[0], 0) + 1

    count_ix[ doc[:-6] ]['total'] = wcount


with open('count_index.json', 'w') as fp:
    json.dump(count_ix, fp)

