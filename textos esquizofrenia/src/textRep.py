from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
import numpy as np
import json
import pickle as pk

class TextRepresentation(object):
    _cv = None
    _TfIdfTransformer = None
    _wc = None
    _tfIdfVec = None
    _idf = None
    _vocab = None
    _binwc = None
    _lbl = None
    _rf_weights = None
    _rf_words = None
    _tfrfl = None
    _binrfl = None
    _tfidf = None
    _binidf = None
    _lblpowerset = False
    _type = "freq"
    _rawData = False
    _dataType = {"freq":"freq","frequency":"freq","numeric":"freq","bin":"bin","binary":"bin","nominal":"bin"}
    def __init__(self,sw = None, no=None, si = False):
        print "TextRepresentation v0.1"
        self._cv = CountVectorizer(stop_words=sw)
        self._TfIdfTransformer = TfidfTransformer(norm=no, smooth_idf=si)
        self._wc = None
        self._tfIdfVec = None
        self._idf = None
        self._vocab = None
        self._binwc = None
        self._lbl = None
        self._rf_weights = None
        self._rf_words = None
        self._tfrfl = None
        self._binrfl = None
        self._tfidf = None
        self._binidf = None
        self._lblpowerset = False
        self._type = "freq"
        self._rawData = False

    #return the dataset
    def getDataSet(self,dataset = "tf-idf"):
        if dataset == "wc":
            return self._wc.copy()
        if dataset == "binwc":
            return self._binwc
        if dataset == "tf-idf":
            return self._tfidf
        if dataset == "tf-rfl":
            return self._tfrfl
        if dataset == "bin-idf":
            return self._binidf
        if dataset == "bin-rfl":
            return self._binrfl

    #get the model type (bin/freq)
    def getType(self):
        return self._type

    #PowerLabelset transformation
    def PLSTransform(self,y):
        aux = map(lambda x: int("".join(map(str, x))), y)
        aux2 = np.zeros(len(aux))
        unique = np.unique(aux)
        for i in range(len(aux)):
            aux2[i] = np.where(unique==aux[i])[0]
        print len(y)
        print len(aux2)
        self._lblpowerset = True
        #Return all the labels transformed,the unique labels and the position of given labels
        return (aux,unique,aux2)

    #creating the count_vectorizer and preprocessing for powerlabel transformation
    def cv_fit(self,arr,lbl,flagLP = False, flagRawData = False, dataType = 'freq'):
        auxlbl = lbl
        self._rawData = flagRawData
        self._lblpowerset = flagLP
        print flagLP
        print self._lblpowerset
        if flagLP:
            print self._lblpowerset
            auxlbl = self.PLSTransform(auxlbl)[2]
        #if it is rawdata, use the countvectorizer, else, copy the data onto the wc/label attributes
        if flagRawData:
            self._cv_fit(arr,auxlbl)
        else:
            self._cv_fit2(arr,auxlbl)
        self._type = self._dataType[dataType]

    def _cv_fit(self, arr, lbl):
        self._wc = self._cv.fit_transform(arr)
        self._vocab = self._cv.vocabulary_
        self._binwc = csr_matrix(np.nan_to_num(np.divide(self._wc, self._wc)))
        self._lbl = np.array(lbl)

    def _cv_fit2(self,arr,lbl):
        self._wc = csr_matrix(arr)
        self._binwc = csr_matrix(np.nan_to_num(np.divide(self._wc, self._wc)))
        self._lbl = np.array(lbl)

    def cv_check(self):
        print self._cv.vocabulary_
        print "====="
        print self._wc.toarray()

    def cv_vocabulary(self):
        return self._vocab

    def trprocess(self):

        #Tfidf fit
        print("==== Fitting TF-IDF ====")
        self._tfidf_fit()
        #binidf
        print("==== Fitting BIN-IDF ====")
        self._binidfFunc()
        print self._lblpowerset
        if self._lblpowerset:
            print("==== Calculating  RFL weights ====")
            self._rf_weight_calc()
            print("==== Fitting TF-RFL ====")
            self._tfrfl_fit()
            print("==== Fitting BIN-RFL ====")
            self._binrfl_fit()
        else:
            print("==== Calculating  RFL weights ====")
            self._rf_weight_calc2()
            print("==== Fitting TF-RFL ====")
            self._tfrfl2_fit()
            print("==== Fitting BIN-RFL ====")
            self._binrfl2_fit()


    def _tfidf_fit(self):
        self._tfIdfVec = self._TfIdfTransformer.fit(self._wc)
        self._idf = self._tfIdfVec._idf_diag
        self._tfidf = self._TfIdfTransformer.transform(self._wc)
        return self._tfidf

    def _binidfFunc(self):
        self._binidf = self._binwc * self._idf
        return self._binidf

    def _rf_weight_calc2(self):
        aux = np.zeros((len(self._lbl[0]),self._wc.shape[1]))
        print aux.shape
        for i in range(len(self._lbl)):
            arr = np.argwhere(self._lbl[i] == 1)
            for el in arr:
                for subele in self._wc[i]:
                    aux[el[0]] = aux[el[0]] + np.nan_to_num(subele[0].sum(axis=0)/subele[0].sum(axis=0))
        self._rf_words = csr_matrix(aux)
        del aux
        #aplying rfl
        aux = self._rf_words.toarray()
        aux_wg = np.zeros(self._rf_words.shape)
        for i in range(0,len(aux)):
            aux_wg[i] = aux[i]
            comp_aux = np.delete(aux,i,axis=0)
            comp_aux = np.maximum(comp_aux.mean(0),1)
            aux_wg[i] = np.log2((aux[i]/comp_aux) + 2)
            del comp_aux
        self._rf_weights = aux_wg.copy()
        del aux_wg
        del aux

    def _rf_weight_calc(self):
        unique_lbls = np.unique(self._lbl)
        aux = np.zeros((len(unique_lbls),self._wc.shape[1]))
        for el in unique_lbls:
            arr = np.argwhere(self._lbl==el)
            for subele in self._wc[arr.T[0]]:
                aux[el] = aux[el] + np.nan_to_num(subele[0].sum(axis=0)/subele[0].sum(axis=0))
        self._rf_words = csr_matrix(aux)
        del aux
        #aplying rfl
        aux = self._rf_words.toarray()
        aux_wg = np.zeros(self._rf_words.shape)
        for i in range(0,len(aux)):
            aux_wg[i] = aux[i]
            comp_aux = np.delete(aux,i,axis=0)
            comp_aux = np.maximum(comp_aux.mean(0),1)
            aux_wg[i] = np.log2((aux[i]/comp_aux) + 2)
            del comp_aux
        self._rf_weights = aux_wg.copy()
        del aux_wg
        del aux
        #print self._rf_weights

    def _tfrfl_fit(self):
        wc_aux = self._wc.toarray().astype(float)
        for i in range(self._wc.shape[0]):
            wc_aux[i] = np.multiply(wc_aux[i], self._rf_weights[self._lbl[i]])
        self._tfrfl = csr_matrix(wc_aux)
        del wc_aux

    def _tfrfl2_fit(self):
        wc_aux = self._wc.toarray().astype(float)
        for i in range(self._wc.shape[0]):
            arr = np.argwhere(self._lbl[i] == 1)
            for el in arr:
                wc_aux[i] = np.multiply(wc_aux[i], self._rf_weights[el[0]])
        self._tfrfl = csr_matrix(wc_aux)
        del wc_aux

    def _binrfl_fit(self):
        bin_aux = self._binwc.toarray().astype(float)
        for i in range(self._binwc.shape[0]):
            bin_aux[i] = np.multiply(bin_aux[i], self._rf_weights[self._lbl[i]])
        self._binrfl = csr_matrix(bin_aux)
        del bin_aux

    def _binrfl2_fit(self):
        bin_aux = self._binwc.toarray().astype(float)
        for i in range(self._binwc.shape[0]):
            arr = np.argwhere(self._lbl[i] == 1)
            for el in arr:
                bin_aux[i] = np.multiply(bin_aux[i], self._rf_weights[el[0]])
        self._binrfl = csr_matrix(bin_aux)
        del bin_aux

    def print_var(self,var ,filename = "file", output="txt"):
        f = open(filename,"w")
        if var == "dictionary":
            out = list(self._vocab.iteritems())
            out.sort()

        print out
        if output == 'json':
            json.dump(out,f)
        if output == 'txt':
            for el in out:

                f.write(el[0].encode('utf8')+":"+str(el[1])+"\n")

        f.close()

    def _printdictionary(self,filename):
        pass

    def exportTR(self,fname, flagFull= False):

        if flagFull:
            f = open(fname, "wb")
            f.write(pk.dumps(self.__dict__,2))
            f.close()
        else:
            f = open(fname+"_wc" , "wb")
            f.write(pk.dumps(self._wc,2))
            f.close()
            f = open(fname+"_lbl" , "wb")
            f.write(pk.dumps(self._lbl,2))
            f.close()
            f = open(fname+"_tfrfl" , "wb")
            f.write(pk.dumps(self._tfrfl,2))
            f.close()
            f = open(fname+"_binrfl" , "wb")
            f.write(pk.dumps(self._binrfl,2))
            f.close()
            f = open(fname+"_vocab" , "wb")
            f.write(pk.dumps(self._vocab,2))
            f.close()
            f = open(fname+"_tfidf" , "wb")
            f.write(pk.dumps(self._tfidf,2))
            f.close()
            f = open(fname+"_idf" , "wb")
            f.write(pk.dumps(self._idf,2))
            f.close()
            f = open(fname+"_binidf" , "wb")
            f.write(pk.dumps(self._binidf,2))
            f.close()
            f = open(fname+"_binwc" , "wb")
            f.write(pk.dumps(self._binwc,2))
            f.close()
            f = open(fname+"_rf_weights" , "wb")
            f.write(pk.dumps(self._rf_weights,2))
            f.close()

    def importTR(self,fname,flagFull = False):
        if flagFull:
            f = open(fname,"r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self.__dict__.update(tmp_dict)
            f.close()
        else:
            f = open(fname+"_wc","r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self._wc = tmp_dict
            f.close()
            f = open(fname+"_lbl","r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self._lbl = tmp_dict
            f.close()
            f = open(fname+"_tfrfl" , "r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self._tfrfl = tmp_dict
            f.close()
            f = open(fname+"_binrfl" , "r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self._binrfl = tmp_dict
            f.close()
            f = open(fname+"_vocab" , "r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self._vocab = tmp_dict
            f.close()
            f = open(fname+"_tfidf" , "r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self._tfidf = tmp_dict
            f.close()
            f = open(fname+"_idf" , "r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self._idf = tmp_dict
            f.close()
            f = open(fname+"_binidf" , "r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self._binidf = tmp_dict
            f.close()
            f = open(fname+"_binwc" , "r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self._binwc = tmp_dict
            f.close()
            f = open(fname+"_rf_weights" , "r")
            conf = f.read()
            tmp_dict = pk.loads(conf)
            self._rf_weights = tmp_dict
            f.close()
