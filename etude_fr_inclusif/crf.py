from ._utils import AnnPredModel, Trie, Ann
import os
import re
import pandas as pd
from unidecode import unidecode
import sklearn_crfsuite
from sklearn.model_selection import train_test_split


# Extracting french dictionnary
with open(os.path.dirname(os.path.abspath(__file__)) + '/data/dictrules.txt') as f : 
	fr_dict = Trie(f.read().split('\n'))


"""
wd : a string
output : the number of characters in wd which are neither lower cas characters nor numerals
"""
def nb_non_alpha(wd):
    cpt = 0
    for car in wd:
        asc = ord(car)
        if not ( (asc >= 65 and asc <= 90) or (asc >= 97 and asc <= 122)) :
            cpt += 1
    return cpt

"""
wd : a string
output : the number of characters in wd which are numerals
"""
def nb_num(wd) :
    cpt = 0
    for car in wd:
        asc = ord(car)
        if (asc >= 48 and asc <= 57) :
            cpt += 1
    return cpt


"""
tk : a spacy Token
output : a vector corresponding to tk
"""
def tk_to_vect(tk) :
	raw = unidecode(tk.text)
	return [ nb_non_alpha(raw), #number of characters which are neither lower case characters nor numerals  
            1 if not fr_dict.exists(raw.lower()) else 0, #1/0 whether the word is present in a french dictionnary
            nb_num(tk.text), #number of numerals
            sum(1 for c in tk.text if c.isupper()), # number of upper case characters 
            len(tk.text), # length of the word
            1 if (tk.pos_ in ['VERB', 'NOUN', 'ADJ']) else 0, # 1/0 whether the word POS tag is among ['VERB', 'ADJ', 'NOUN'], or not
            abs(len(tk.text) - len(tk.lemma_)) ] # gap between the length of the text and the length of its stem

"""
lv : a list of vector (in the form discribed in tk_to_vect)
output : a list of dict, translated from lv, in such a way that the crf can process it
"""
def vects_to_features(lv) :
    res = []
    for i in range(0,len(lv)):
        v = lv[i][:7]
        nv = {"nbnonalpha" : v[0] , "indict" : v[1] == 1 , "nbnum" : v[2] , "nbmaj" : v[3] , "length" : v[4] , "pos" : v[5] == 1 , "gaplengthstem" : v[6]}
        for j in range(-2,+3,1):
            k = i + j
            if k >= 0 and k < len(lv) :
                nv["val" + ("+" if j >= 0 else "") + str(j)] = str(lv[k])
            else:
                nv["val" + ("+" if j >= 0 else "") + str(j)] = ""  
        #nv["-1to0"] = str(lv[i-1])  + str(lv[i]) if i > 0 else ""
        #nv["0to1"] = str(lv[i]) + str(lv[i+1]) if i < len(lv) - 1 else ""
        #nv["-1to1"] = str(lv[i-1]) + str(lv[i]) if 0 < i < len(lv) - 1 else ""
            
        res.append(nv)
    return res
"""
tks : an iterable of Token
output : a list of dict, translated from tks, in such a way that the crf can process it
"""
def tks_to_features(tks):
	return vects_to_features([tk_to_vect(tk) for tk in tks])


"""
class of annotation crf-based model
extends AnnPredModel
"""
class CRFModel(AnnPredModel) :

	def __init__(self, labelmethod='IO') :
		super().__init__(labelmethod=labelmethod)
		self.crf_classif = sklearn_crfsuite.CRF( algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=1000, all_possible_transitions=True)

	"""
	text : a document
	anns : a list of annotations in this document
	output : reshape the data (text) in form of a DataFrame with the following columns : [texts, tokens, annotated, labels] . Each row corresponds to a document (so there is only one row)
	"""
	def _sub_crf_shaping(self, text : str, anns : list[Ann]) -> pd.DataFrame :
		doc = self.nlp_model(text)
		dfdict = {"texts" : [[]], "tokens" : [[]], "feats": [], "annotated" : [[]], "labels" : [[]]}
		for tk in doc:
			dfdict["texts"][0].append(tk.text)
			dfdict["tokens"][0].append(tk)
			indcpl = (tk.idx, tk.idx + len(tk.text))
			b = False
			for ann in anns :
				b = b or indcpl in ann
			if not b :
				dfdict["labels"][0].append('O')
			elif self.labelmethod == 'IOB' and (len(dfdict["annotated"][-1]) == 0 or not dfdict["annotated"][-1][-1] ) :
				dfdict["labels"][0].append('B')
			else:
				dfdict["labels"][0].append('I')
			dfdict["annotated"][0].append(b)
		dfdict["feats"].append(tks_to_features(dfdict["tokens"][0]) )
		return pd.DataFrame(dfdict)

	"""
	texts : a list of document
	lanns : a list of list of annotations (annotations from the i-th sublist are coming from the i-th element from texts)
	output : reshape the data (texts) in form of a DataFrame with the following columns : [texts, tokens, annotated, labels] . Each row corresponds to a document
	"""
	def _crf_shaping(self, texts : list[str], lanns : list[list[Ann]]) -> pd.DataFrame :
		df = pd.DataFrame({"texts" : [], "tokens" : [], "feats": [], "annotateds" : [], "labels" : []})
		for i in range(0, len(texts)) :
			df = pd.concat([df, self._sub_crf_shaping(texts[i], lanns[i])], ignore_index=True) 
		return df


	def fit(self, x : list[str], y : list[list[Ann]]) -> None :
		df = self._crf_shaping(x, y)
		self.crf_classif = self.crf_classif.fit(df["feats"], df["labels"])

	def _predict(self, x : list[str]) -> list[list[Ann]] :
		res = []
		for text in x :
			res.append([])
			df = self._crf_shaping([text], [[]])
			preds = self.crf_classif.predict( df["feats"] )
			inproc = False
			for i in range(0, len(preds)) :
				pred = preds[i]
				for j in range(0, len(pred)):
					b = (pred[j] != 'O')
					if (not inproc) and b :
						beg = df["tokens"][i][j].idx
					elif inproc and not b :
						last = df["tokens"][i][j - 1]
						end = last.idx + len(last.text) 
						res[-1].append(Ann(beg, end, text=text[beg:end] ) )
					inproc = b
			if inproc :
				res[-1].append(Ann(beg, len(text), text=text[beg:] ))
		return res
			 	


