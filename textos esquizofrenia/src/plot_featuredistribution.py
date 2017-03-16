import numpy as np
import json
import matplotlib.pyplot as plt

# Load the inverted index with the features and the freqs:
with open('inv_index.json', 'r') as fp:
    inv_ix = json.load(fp)

sorted_tags = np.sort(inv_ix.keys())

dclass = np.array([int(l.strip().split(" ")[1]) for l in open('classes')])
dnames = np.array([l.strip().split(" ")[0] for l in open('classes')])


doc_counts = json.load(open('count_index.json'))

negative_counts = {}
positive_counts = {}

positive_docs = np.sort(dnames[np.where(dclass == 1)])
negative_docs = np.sort(dnames[np.where(dclass == -1)])


for doc in positive_docs:
    total = float(doc_counts[doc]['total'])
    for tag in doc_counts[doc]:
        if tag != 'total':
            if not tag in positive_counts:
                positive_counts[tag] = []

            positive_counts[tag].append(doc_counts[doc][tag] / total)


for doc in negative_docs:
    total = float(doc_counts[doc]['total'])
    for tag in doc_counts[doc]:
        if tag != 'total':
            if not tag in negative_counts:
                negative_counts[tag] = []

            negative_counts[tag].append(doc_counts[doc][tag] / total)


pos_count_dist = {'positive':positive_counts, 'negative':negative_counts}

with open('pos_distribution.json', 'w') as fp:
    json.dump(pos_count_dist, fp)

import matplotlib.pyplot as plt

pos_names = dict(zip([u'A', u'C', u'D', u'F', u'I', u'N', u'P', u'R', u'S', u'V', u'W', u'Z'], [u'adjective', u'conjunction', u'determiner', u'punctuation', u'interjection', u'noun', u'pronoun', u'adverb', u'adposition', u'verb', u'date', u'number']))


# removing punctuation
if 'F' in positive_counts:
    del positive_counts['F']

if 'F' in negative_counts:
    del negative_counts['F']


sorted_pos_tags = np.sort(positive_counts.keys())
sorted_neg_tags = np.sort(negative_counts.keys())

data_pos = []
for tag in sorted_pos_tags:
    data_pos.append(positive_counts[tag])

data_neg = []
for tag in sorted_neg_tags:
    data_neg.append(negative_counts[tag])







f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.boxplot(data_pos)
ax1.set_title('Non-Schizo')
ax1.set_xticklabels([pos_names[x] for x in sorted_pos_tags], rotation='vertical')
ax1.grid()

ax2.boxplot(data_neg)
ax2.set_title('Schizo')
ax2.set_xticklabels([pos_names[x] for x in sorted_neg_tags], rotation='vertical')
ax2.grid()
plt.suptitle('Portion of tag occurrences in each document')



#ax.legend((rects1[0], rects2[0]), ('Positive', 'Negative'))
#ax.set_xticklabels(new_tags)
#ax.set_xticklabels([pos_names[x] for x in new_tags], rotation='vertical')
#ax.set_xticks(ind + width / 2)
#fig.subplots_adjust(bottom=0.18)
plt.show()
#plt.savefig("post_count.png")


'''
tag_positive_counts = []
tag_negative_counts = []

positive_docs = set(dnames[np.where(dclass == 1)])
negative_docs = set(dnames[np.where(dclass == -1)])

##########
#### OPT 1: Grouping by POS
##########
new_tags = []
for t in sorted_tags:
    if len(new_tags) == 0:
        new_tags.append(t[0])
        continue
    if new_tags[-1] != t[0]:
        new_tags.append(t[0])


index = 0
for tag in sorted_tags:
    if tag[0] == new_tags[index]:
        if len(tag_positive_counts) == 0:
            tag_positive_counts.append(0)
            tag_negative_counts.append(0)
        tag_positive_counts[index] += len(set(inv_ix[tag].keys()) & positive_docs)
        tag_negative_counts[index] += len(set(inv_ix[tag].keys()) & negative_docs)
    else:
        index += 1
        tag_positive_counts.append(len(set(inv_ix[tag].keys()) & positive_docs) )
        tag_negative_counts.append(len(set(inv_ix[tag].keys()) & negative_docs) )





pos_names = dict(zip([u'A', u'C', u'D', u'F', u'I', u'N', u'P', u'R', u'S', u'V', u'W', u'Z'], [u'adjective', u'conjunction', u'determiner', u'punctuation', u'interjection', u'noun', u'pronoun', u'adverb', u'adposition', u'verb', u'date', u'number']))

tag_positive_counts = 100 * np.array(tag_positive_counts) / max(tag_positive_counts)
tag_negative_counts = 100 * np.array(tag_negative_counts) / max(tag_negative_counts)

width = 0.35
#ind = np.arange(len(sorted_tags))
ind = np.arange(len(new_tags))



fig, ax = plt.subplots()
rects1 = ax.bar(ind, tag_positive_counts, width, color='r')
rects2 = ax.bar(ind + width, tag_negative_counts, width, color='y')
ax.set_title('POS tag doc-count per class (No-Schizo vs Schizo')
ax.legend((rects1[0], rects2[0]), ('Positive', 'Negative'))
#ax.set_xticklabels(new_tags)
ax.set_xticklabels([pos_names[x] for x in new_tags], rotation='vertical')
ax.set_xticks(ind + width / 2)
fig.subplots_adjust(bottom=0.18)
#plt.show()
plt.savefig("post_count.png")

'''