#coding: utf-8
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from textRep import TextRepresentation
import os

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

relato = "Relato C" # Cambiar por relato necesario

flagIdf = True
x_df = pd.read_csv("Parsed/"+relato+".csv")


np.set_printoptions(threshold=np.nan)

inv = x_df.ix[:,:4]
data = x_df["rawtext"]
lbl = np.array(x_df["is_ezq"])

data2 = np.array(data)
for i in range(len(data2)):
    data2[i] = (data2[i]).decode('utf8')

tr = TextRepresentation(sw=None,no=None,si=True) # sw -> StopWords=None, no -> Normalization = None,  Smooth-idf = True
tr.cv_fit(data2,lbl,flagLP=False,flagRawData=True,dataType='freq') #No utilizar LabelPowerset, Es data en bruto (Palabras), tipo de representacion Frecuencias
tr._tfidf_fit()

x = tr._vocab.keys()
x.sort()

f = open("tf-idf/"+relato+".csv","w")
f.write("id,filename,id_user,is_ezq,")
for el in x:
    f.write(el.encode('utf8')+",")
f.write("\n")

if flagIdf:
    arr = tr._tfidf
else:
    arr = tr._wc
for i in range(tr._wc.shape[0]):
    f.write(str(inv["id"][i])+","+str(inv["filename"][i])+","+str(inv["id_user"][i])+","+str(inv["is_ezq"][i])+","+np.array2string(arr[i].toarray()[0],separator=',').replace('\n','').replace(' ','')[1:-1]+"\n")
f.close()


if flagIdf:
    ensure_dir("dict/tf-idf/")
    tr.print_var("dictionary","dict/tf-idf/"+relato+".txt","txt")
else:
    ensure_dir("dict/freq/")
    tr.print_var("dictionary","dict/freq/"+relato+".txt","txt")
ensure_dir("Models/")

tr.exportTR("Models/"+relato,True)