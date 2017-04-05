# author: Juan Zamora

import json
import sys
import optparse
import string

'''
Aplicada en cada carpeta como:
for i in $( ls *.ana ); do python ../../../src/postagging.py -i $i; done;
'''

parser = optparse.OptionParser()
parser.add_option('-i', '--input', dest='input', help='Input ana file')
parser.add_option('-o', '--output', dest='output', help='Output txt file where the token pairs will be written')

(options, args) = parser.parse_args()

if options.input is None:
    print("Usage:"+sys.argv[0]+" -i inputfile_path [-o outputfilepath]")
    sys.exit(-1)

if options.output is None:
    options.output = options.input[:-3] + "pairs"


cc = json.load(open(options.input))
out = open(options.output, 'w')

tags_found = set()

for paragraph in cc['paragraphs']: # each item is a dict with key -> 'sentences'
    for sentence in paragraph['sentences']:# each item is a dict with keys ->'tokens' (list with tokens) , 'id' (sentence num)
        for token in sentence['tokens']:# each item is a dict with keys -> 'ctag', 'form', 'pos', 'lemma', 'tag', 'id'
            token_info = token['form'], token['tag']
            if not token_info[0] in string.punctuation:
                tags_found.add(token['tag'])
                out.write('{0}:{1}\n'.format(token_info[0].encode('utf-8'), token_info[1]) )



out.close()

tagsf = open('tags_found', 'a')
for t in tags_found:
    tagsf.write(t+"\n")

tagsf.close()

#print "File "+options.output+" succesfully generated (punctuation removed)."