from ._utils import AnnPredModel, Ann, Trie, group, merge_sort
import os
import Levenshtein
from unidecode import unidecode
import re

# regex to identify characters which are not alphaetics or numerals
punct = "[^a-zA-Z0-9çàâéèêëîïôöûüæœ]"

# regex to identify french words whose plural form is not singular +  s
excpt = re.compile(r'.*(s|x|z|al|ail|ou|eau|au|eu)$')

# regex for standard french plural suffix
plur =  "(s|" + punct + "s)?"

parent_path = os.path.dirname(os.path.dirname(__file__) )

"""
class of data structure inspired from Trie but for regular expressions
only splits on characters which are alphabetics or numerals
"""
class SortedRegex :

	"""
	lv : level of the node
	"""
	def __init__(self, lv=0) :
		self.__nbitems = 0
		self.vals = [] # list of regex
		self.children = {}
		self.lv = lv
	
	"""
	item : the string added to the SortedRegex
	itemid : the id of this item 
	"""
	def __add(self, item : str, itemid) :
		if len(item) <= self.lv  or  re.match(punct, item[self.lv]):
			self.vals.append({"regexp" : item, "id" : itemid})
		else:
			car = item[self.lv]
			if car not in self.children :
				self.children[car] = SortedRegex(lv= self.lv + 1)
			self.children[car].__add(item, itemid)
	"""
	item : the string added to the SortedRegex
	itemid : the id of this item (default : the number of items in this SortedRegex)
	"""
	def add(self, item, itemid=None) :
		self.__nbitems += 1
		self.__add(item, itemid if itemid is not None else self.__nbitems)

	"""
	item : a string
	output : the id of the first regular expression to match with item found in this SortedRegex, or None if no one found
	"""
	def match(self, item) :
		for val in self.vals :
			if re.match(val["regexp"], item) :
				return val["id"]
		if len(item) <= self.lv or re.match(punct, item[self.lv]) or item[self.lv] not in self.children:
			return None
		else :
			return self.children[item[self.lv]].match(item)

# Extract grammatical data

rules = open(parent_path + "/data/flexrulesuni.txt", 'r')
lines = rules.readlines()
uniflexrulesregex = SortedRegex()
unicoordrulesregex = SortedRegex()
idcpt = 0
for line in lines:
	idcpt += 1
	parts = line.split('\t')
	uniflexrulesregex.add(parts[2][:-1], idcpt)
	unicoordrulesregex.add( r'' + parts[0] + 's?' + parts[1] + 's?', idcpt)
rules.close()

rules = open(parent_path + "/data/epicenerules.txt", 'r')
lines = rules.readlines()
epirulesregex = Trie([line[:-1] for line in lines])
rules.close()

rules = open(parent_path + "/data/listetermesneutres.txt", 'r')
lines = rules.readlines()
neutrulesregex = Trie([line[:-1] for line in lines])
rules.close()

rules = open(parent_path + "/data/dictrules.txt", 'r')
lines = rules.readlines()
dictrules = Trie([line[:-1] for line in lines])
rules.close()


# dectection functions

"""
wd : a word
output : a boolean indicating whether wd is an inflexion
"""
def detect_flex(wd : str):
	return uniflexrulesregex.match(wd.lower())

"""
wd : a word
output : a boolean indicating whether wd is epicene
"""
def detect_epi(wd : str):
	return -1 if epirulesregex.exists(wd.lower()) else None

"""
wd : a word
output : a boolean indicating whether wd is a neutral-gender word
"""
def detect_neut(wd : str):
	if neutrulesregex.exists(wd) :
		return -1

"""
wd : a word
output : the list of the possible masculine inflections for this word
"""
def masc_inf(wd: str) :
	if re.match(r'.*ettes?$', wd) :
		return [re.sub(r'ettes?$', 'et', wd), re.sub(r'ettes?$', 'e', wd), re.sub(r'ettes?$', '', wd)]
	if re.match(r'.*esses?$', wd) :
		return [re.sub(r'esses?$', 'e', wd), re.sub(r'esses?$', '', wd)]
	if re.match(r'.*ères?$', wd) :
		return [ re.sub(r'ères?$', 'er', wd) ]
	if re.match(r'.*(r?ice|eure|euse)s?$', wd) :
		return [ re.sub(r'(r?ice|eure|euse)s?$', 'eur' ,wd) ]
	if re.match(r'.*effes?$', wd) :
		return [ re.sub(r'effes?$', 'ef', wd) ]
	if re.match(r'.*inn?es?$', wd) :
		return [re.sub(r'inn?es?$', 'in', wd), re.sub(r'inn?es?$', 'ain', wd) ]
	if re.match(r'.*es?$', wd) :
		return [re.sub(r'es?$', '', wd) ]
	return None



"""
wd : a word
output : a boolean indicating whether wd is a feminisation
"""
def detect_fem(wd : str):
	exc = ["France", "france"]
	masc = masc_inf(wd)
	if len(wd) < 6 or masc is None or wd in exc :
		return None
	else:
		if not dictrules.exists( wd.lower() ) and sum(1 if dictrules.exists( m.lower() ) else 0 for m in masc ) > 0 :
			return -1
		else:
			return None

"""
tokens : a list of Tokens (from a spacy model)
output : a boolean indicating whether tks contains a coordination process
"""
def detect_coord(tks):
	for i in range(0, len(tks)):
		for j in range(i+1, len(tks)):
			itk = tks[i]
			jtk = tks[j]
			remain = [tks[k] for k in range(0, len(tks)) if k not in [i,j] ]
			if 'CCONJ' in [item.pos_ for item in remain] or 'PUNCT' in [item.pos_ for item in remain]:
				tA = unicoordrulesregex.match(itk.text + jtk.text)
				if tA is not None:
					return Ann(itk.idx, jtk.idx + len(jtk.text), metadata={"category" : ["coo"]} )
				tB = unicoordrulesregex.match(jtk.text + itk.text)
				if tB is not None:
					return Ann(itk.idx, jtk.idx + len(jtk.text), metadata={"category" : ["coo"]})
	return None

"""
doc : a spacy Document
coord_range : the number of consecutive words to check in doc when detecting coordinations
proc : the list of processes to detect in documents among the following : 'fle' for inflections, 'neu' for neutral-gender, 'fem' for feminisation, 'epi' for epicene words
output : a list of Ann (annotations in the document)
"""
def detect_inc(doc, coord_range=4, proc=['fle', 'neu', 'fem', 'epi', 'coo']):
	res = []
	if 'coo' in proc :
		tks_grps = group([tk for tk in doc], coord_range)
		for gr in tks_grps:
			tmp = detect_coord(gr)
			if tmp is not None :
				res.append([tmp])
	for tk in doc :
		if 'fle' in proc :
			tmp = detect_flex(tk.text)
			if tmp is not None:
				res.append([Ann(tk.idx, tk.idx + len(tk.text), text=tk.text, metadata={"category" : ["fle"]}) ])
		if 'neu' in proc :
			tmp = detect_neut(tk.text)
			if tmp is not None:
				res.append([Ann(tk.idx, tk.idx + len(tk.text), text=tk.text, metadata={"category" : ["neu"]}) ])
		if 'fem' in proc :
			tmp = detect_fem(tk.text)
			if tmp is not None:
				res.append([Ann(tk.idx, tk.idx + len(tk.text), text=tk.text, metadata={"category" : ["fem"]}) ])
		if 'epi' in proc :
			tmp = detect_epi(tk.text)
			if tmp is not None:
				res.append([Ann(tk.idx, tk.idx + len(tk.text), text=tk.text, metadata={"category" : ["epi"]}) ])
	return merge_sort(res)


"""
class of annotation advanced rule-based model
extends AnnPredModel
"""
class AdvRBModel(AnnPredModel) :

	"""
	proc : the list of processes to detect in documents among the following : 'fle' for inflections, 'neu' for neutral-gender, 'fem' for feminisation, 'epi' for epicene words
	"""
	def __init__(self, proc=['fle', 'neu', 'fem', 'epi', 'coo']):
		super().__init__()
		self.proc = proc

	def fit(self, x : list[str], y : list[list[Ann]]) -> None :
		print("This model doesn't need any kind of training : the 'fit' function is not doing anything")

	def _predict(self, x : list[str]) -> list[list[Ann]] :
		return [ detect_inc(self.nlp_model(item), proc=self.proc) for item in x ]
