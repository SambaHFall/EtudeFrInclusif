from abc import ABC, abstractmethod
import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_suffix_regex


"""
nlp : spacy model
output : new Tokenizer more adjested to inclusive french
"""
def change_tokenizer(nlp) :
    inf = list(nlp.Defaults.infixes)
    infixes = [x for x in inf if '-|–|—|--|---|——|~' not in x ] # remove rule that separate words on hyphens
    infix_re = compile_infix_regex(infixes)
    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

"""
loadfrom : name of the spacy model
output : spacy model associated to loadfrom, adjusted to inclusive french
"""
def spacynlp(loadfrom : str) :
	nlp = spacy.load(loadfrom)
	nlp.tokenizer = change_tokenizer(nlp)
	return nlp

def retokenization(doc) :
	modif = False
	prec = None
	for tk in doc :
		if tk.text == ')' and prec is not None and '(' in prec.text:
			with doc.retokenize() as retokenizer :
				retokenizer.merge(doc[(tk.i - 1) : (tk.i + 1)] )
			modif = True
			break
		prec = tk

	if modif :
		return retokenization(doc)
	else :
		return doc

"""
output : training data load from "corpus français inclusif" in form of : list[str], list[list[Ann]]  
"""
def get_fr_inclusif_data():
	df = pd.read_csv("data/fr_inclusif.csv")
	return [item for item in df["Text"]], [ [Ann(item["beg"], item["end"], text=item["text"], metadata=item["metadata"]) for item in eval(jtem)] for jtem in df["Annotations"]  ]


"""
l : list
n : number
output : list of every n-sized sublists of consecutive element from l 
"""
def group(l, n) :
    res = [[item] for item in l]
    for i in range(1,n):
        for ind in range(i,len(l)) :
            res[ind - i].append(l[ind])
    return res


"""
Node of a Trie
Each node is associated to a char
A path from the root to any other node is associated to a word (formed by concatenating every char associated to every node through the path)
"""
class TrieNode:

	"""
	c : char associated to this node
	"""
	def __init__(self, c : str) :
		self.char = c
		self.children = {} # contains children
		self.exbit = False # indicates whether the word associated to the path coming from the root to this node exists or not

	"""
	suff : word added to the Trie
	"""
	def add(self, suff : str) -> None :
		if len(suff) > 0 :
			c = suff[0]
			if c not in self.children :
				self.children[c] = TrieNode(c)
			self.children[c].add(suff[1:])
		else:
			self.exbit = True

	"""
	wd : a word 
	output : a boolean indicating wheter wd exists in the Trie
	"""
	def exists(self, wd : str) -> bool :
		if len(wd) == 0 :
			return self.exbit
		else :
			return wd[0] in self.children and self.children[wd[0]].exists(wd[1:])


"""
Trie class (actually the root of a Trie)
"""
class Trie(TrieNode) :


	"""
	loadfrom : list of strings added to the Trie
	"""
	def __init__(self, loadfrom : list[str]) :
		super().__init__('') # the root is arbitrarily associted to ''
		for wd in loadfrom :
			self.add(wd)



"""
class representing an annotation in a document
basically, an annotation is identified by two indexes : one marking where the annotation start (included), and the another one marking where it ends (excluded)
"""
class Ann:

	"""
	beg : the index where the annotation starts in the document
	end : teh index where the annotation ends in the document (actually the first index after the annotation)
	text : the part of the document which is annotated
	metadata : a dict containing informations about the annotation
	"""
	def __init__(self, beg : int, end : int, text=None, metadata={}):
		self.beg = beg
		self.end = end
		self.text = text
		self.metadata = metadata

	def __str__(self) :
		return "[" + str(self.beg) + ":" + str(self.end) + "] -> " + ("?" if self.text is None else self.text) + ("" if self.metadata is None else " || " + str(self.metadata))  

	def __repr__(self) :
		return "[" + str(self.beg) + ":" + str(self.end) + "] -> " + ("?" if self.text is None else self.text) + ("" if self.metadata is None else " || " + str(self.metadata))  

	"""
	a : can be either a Ann or a couple of indexes
	output : a boolean indicating if a is included within this annotation
	"""
	def __contains__(self, a) -> bool :
		if isinstance(a, Ann) :
			indcpl = (a.beg, a.end)
		else :
			indcpl = a
		return self.beg <= indcpl[0] < self.end and self.beg < indcpl[1] <= self.end


"""
anns_a, anns_b : lists of Ann
output : merges and sorts the two lists
(if an annotation from one list is included within a annotation coming from the other one, it is removed)
"""
def sub_merge_sort(anns_a : list[Ann], anns_b : list[Ann]) ->list[Ann] :
	res_a = []
	res_b = []

	for ann_a in anns_a :
		sup = None
		for ann_b in anns_b :
			if ann_a in ann_b :
				sup = ann_b
		if sup is None :
			res_a.append(ann_a)
		else :	
			if "category" in sup.metadata and "category" in ann_a.metadata :
				sup.metadata["category"] += ann_a.metadata["category"]

	for ann_b in anns_b :
		sup = None
		for ann_a in res_a :
			if ann_b in ann_a :
				sup = ann_a
		if sup is None :
			res_b.append(ann_b)
		else :
			sup.metadata["category"] += ann_b.metadata["category"]

	return sorted( res_a + res_b , key = lambda it : it.beg )

"""
lanns : a list of list of Ann
output : sorted list of Ann which results from merging all the sublists of lanns
"""
def merge_sort(lanns : list[list[Ann]]) -> list[Ann] :
	if len(lanns) == 0 :
		return []
	elif len(lanns) == 1 :
		return lanns[0]
	elif len(lanns) == 2 :
		return sub_merge_sort(lanns[0], lanns[1])
	else :
		return sub_merge_sort(lanns[0], merge_sort(lanns[1:]))

"""
Abstarct class representing an annotation model
"""
class AnnPredModel(ABC):

	"""
	nlpmodel : name of the nlp model used by the annotation model
	labelmethod : method of labeling, either 'IO' or 'IOB' ('IO' by default)
	"""
	def __init__(self, nlpmodel="fr_core_news_sm", labelmethod='IO') :
		self.nlp = spacynlp(nlpmodel)
		self.labelmethod = labelmethod

	""" 
	x : a list of documents
	y : a list of list of Ann
	the i-th element from y contains all the annotations in the i-th element if x
	"""
	@abstractmethod
	def fit(self, x : list[str], y : list[list[Ann]]) -> None :
		pass



	""" 
	x : a list of documents
	output : a list of list of Ann
	the i-th element from the output contains all the annotations predicted in the i-th element if x
	"""
	@abstractmethod
	def predict(self, x : list[str]) -> list[list[Ann]] :
		pass


	def nlp_model(self, text : str) :
		doc = self.nlp(text)
		return retokenization(doc)

	""" 
	text : a document
	anns : a list of annotations in this document
	offset : number of chars to keep before and after the annotations in order to capture the context
	iddoc : the id of the document
	output : reshape the data (text and its annotations) in form of a DataFrame with the following columns : "iddoc", "context", "range", "text"
	"""
	def _sub_annotation_layout(self, text : str, anns : list[Ann], offset=40, iddoc=None) -> pd.DataFrame :
		dfdict = { "iddoc" : [], "context" : [], "range" : [], "text" : [], "category" : []}
		for ann in anns :
			dfdict["iddoc"].append(iddoc)
			if offset is None :
				beg = 0
				end = len(text)
			else :
				beg = max(ann.beg - offset, 0)
				end = min(ann.end + offset, len(text))
			dfdict["context"].append(text[beg:end])
			dfdict["text"].append(text[ann.beg : ann.end])
			dfdict["range"].append({'beg' : ann.beg, 'end' : ann.end})
			dfdict["category"].append(ann.metadata["category"] if "category" in ann.metadata else 'UNKNOWN')
		return pd.DataFrame(dfdict)

	""" 
	texts : a list of document
	lanns : a list of list of annotations (annotations from the i-th sublist are coming from the i-th element from texts)
	offset : number of chars to keep before and after the annotations in order to capture the context
	iddoc : a list contaning the id of the documents (must have the same length as texts and lanns)
	output : reshape the data (texts and their annotations) in form of a DataFrame with the following columns : "iddoc", "context", "range", "text" (annotations are not grouped by document)
	"""
	def annotation_layout(self, texts : list[str], lanns : list[list[Ann]], offset=40, iddoc=[]) -> pd.DataFrame :
		df = pd.DataFrame( [], columns=["iddoc", "context", "range", "text", "category"])
		for i in range(0, len(texts)) :
			df = pd.concat([df, self._sub_annotation_layout(texts[i], lanns[i], offset=offset, iddoc = None if len(iddoc) == 0 else iddoc[i])], ignore_index=True)
		return df

	"""
	text : a document
	anns : a list of annotations in this document
	output : reshape the data (text) in form of a DataFrame with the following columns : [texts, tokens, annotated, labels] . Each row corresponds to a token of the text
	"""
	def _sub_shaping(self, text : str, anns : list[Ann]) -> pd.DataFrame :
		doc = self.nlp_model(text)
		dfdict = {"texts" : [], "tokens" : [], "annotated" : [], "labels" : [], "categories" : []}
		for tk in doc :
			dfdict["texts"].append(tk.text)
			dfdict["tokens"].append(tk)
			indcpl = (tk.idx, tk.idx + len(tk.text))
			cpt = 0
			while cpt < len(anns) and indcpl not in anns[cpt]:
				cpt += 1
			b = cpt < len(anns)
			if not b :
				dfdict["labels"].append('O')
			elif self.labelmethod == 'IOB' and (len(dfdict["annotated"][-1]) == 0 or not dfdict["annotated"][-1][-1]) :
				dfdict["labels"].append('B')
			else:
				dfdict["labels"].append('I')
			dfdict["annotated"].append(b)
			dfdict["categories"].append( anns[cpt].metadata["category"] if b and "category" in anns[cpt].metadata else [] )
		return pd.DataFrame(dfdict)

	"""
	texts : a list of document
	lanns : a list of list of annotations (annotations from the i-th sublist are coming from the i-th element from texts)
	output : reshape the data (texts) in form of a DataFrame with the following columns : [texts, tokens, annotated, labels] . Each row corresponds to a token from one the text (token are not grouped by document)
	"""
	def _shaping(self, texts : list[str], lanns : list[list[Ann]]) -> pd.DataFrame :
		df = pd.DataFrame({"texts" : [], "tokens" : [], "annotateds" : [], "labels" : [], "categories" : []})
		for i in range(0, len(texts)) :
			df = pd.concat([df, self._sub_shaping(texts[i], lanns[i])], ignore_index=True) 
		return df


	"""
	texts : a list of document
	obs : a list of list of annotations observed in the document (annotations from the i-th sublist are coming from the i-th element from texts)
	pred : a list of list of annotations predicted (annotations from the i-th sublist are coming from the i-th element from texts)
	"""
	def metric(self, texts: list[str], obs : list[list[Ann]], pred : list[list[Ann]]) :
		dfobs = self._shaping(texts, obs)
		dfpred = self._shaping(texts, pred)
		nbobs = sum(1 if row["annotated"] else 0 for i, row in dfobs.iterrows())
		nbpred = sum(1 if row["annotated"] else 0 for i, row in dfpred.iterrows())
		nbtruepos = sum(1 if dfobs["annotated"][i] and dfpred["annotated"][i] else 0 for i in range(0, dfobs.shape[0]) )
		prec = 0 if nbpred == 0 else nbtruepos / nbpred
		rec = 0 if nbobs == 0 else nbtruepos / nbobs
		fscore = None if prec + rec == 0 else (2 * prec * rec) / (prec + rec)
		return {"nbobs" : nbobs, "nbpred " :  nbpred, "nbtruepos" : nbtruepos, "precision" : None if nbpred == 0 else prec, "recall" : None if nbobs == 0 else rec, "f1-score" : fscore}
