import docx2txt
import os

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


basedir = "Relatos Esquizofrenia/"

file = "Relatos Esquizofrenia/Relato A/Control/1.a.docx"
test = docx2txt.process(file)


for subdir, dirs, files in os.walk(basedir):
    for file in files:

        filedir = os.path.join(subdir,file)
        output = docx2txt.process(filedir)
        split = filedir.split("/")[1]
        split = split.split("\\")
        print split
        ensure_dir("txts\\"+split[0]+"\\")
        f = open("txts\\"+split[0]+"\\"+split[1]+split[2].split('.')[0]+".txt","w")
        f.write(output.encode("utf8"))
        f.close()