import os
import docx2txt
import numpy as np
import re

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

baseDir = "./txts/Relato A/"
outDir = "Parsed/"
print baseDir
x = []
pat = re.compile("[-.0-9]+")

for subdir, dirs, files in os.walk(baseDir):
    for file in files:

        filedir = os.path.join(subdir,file)
        #print filedir
        f = open(filedir,"r")
        line = f.readlines()
        if pat.match(line[0]):
            line = line[2]
            cat = 1
            iduser = file[:-4]
        else:
            f.seek(0)
            line = f.read()
            cat = 0
            iduser = file[:-5]

        print iduser
        #line = f.read()
        x.append((file,iduser,cat,unicode(line, encoding='utf8', errors = 'replace').replace("\n"," ").replace(",",";")))
        f.close()

x = np.array(x)

f = open(outDir+baseDir.split("/")[2]+".csv","w")
f.write("id,filename,id_user,is_ezq,rawtext\n")
for i,el in enumerate(x):
    f.write(str(i)+","+el[0].encode('utf8')+","+str(el[1])+","+str(el[2])+","+el[3].encode('utf8')+"\n")
f.close()