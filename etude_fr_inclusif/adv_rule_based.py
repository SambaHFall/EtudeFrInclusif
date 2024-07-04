from ._utils import AnnPredModel, Ann, Trie, group, merge_sort
import os
from unidecode import unidecode
import re

"""
class of data structure inspired from Trie but for regular expressions
only splits on characters which are alphabetics or numerals
"""
class SortedRegex:

	"""
	lv : level of the node
	"""
	def __init__(self, lv=0) :
		self.__nbitems = 0
		self.vals = [] # list of regex
		self.children = {}
		self.lv = lv
		# regex to identify characters which are not alphaetics or numerals
		self.punct = "[^a-zA-Z0-9çàâéèêëîïôöûüæœ]"
	
	"""
	item : the string added to the SortedRegex
	itemid : the id of this item
	"""
	def __add(self, item : str, itemid) :
		if len(item) <= self.lv  or  re.match(self.punct, item[self.lv]):
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
		if len(item) <= self.lv or re.match(self.punct, item[self.lv]) or item[self.lv] not in self.children:
			return None
		else :
			return self.children[item[self.lv]].match(item)


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

		# Extract grammatical data
		rules = open(os.path.dirname(os.path.abspath(__file__)) +  "/data/flexrulesuni.txt", 'r')
		lines = rules.readlines()
		self.uniflexrulesregex = SortedRegex()
		self.unicoordrulesregex = SortedRegex()
		idcpt = 0
		for line in lines:
			idcpt += 1
			parts = line.split('\t')
			self.uniflexrulesregex.add(parts[2][:-1], idcpt)
			self.unicoordrulesregex.add( r'' + parts[0] + 's?' + parts[1] + 's?', idcpt)
		rules.close()

		rules = open(os.path.dirname(os.path.abspath(__file__)) + "/data/epicenerules.txt", 'r')
		lines = rules.readlines()
		self.epirulesregex = Trie([line[:-1] for line in lines])
		rules.close()

		rules = open(os.path.dirname(os.path.abspath(__file__)) +  "/data/listetermesneutres.txt", 'r')
		lines = rules.readlines()
		self.neutrulesregex = Trie([line[:-1] for line in lines])
		rules.close()

		rules = open(os.path.dirname(os.path.abspath(__file__)) +  "/data/dictrules.txt", 'r')
		lines = rules.readlines()
		self.dictrules = Trie([line[:-1] for line in lines])
		rules.close()


	"""
	wd : a word
	output : a boolean indicating whether wd is an inflexion
	"""
	def detect_flex(self, wd : str):
		return self.uniflexrulesregex.match(wd.lower())

	"""
	wd : a word
	output : a boolean indicating whether wd is epicene
	"""
	def detect_epi(self, wd : str):
		return -1 if self.epirulesregex.exists(wd.lower()) else None

	"""
	wd : a word
	output : a boolean indicating whether wd is a neutral-gender word
	"""
	def detect_neut(self, wd : str):
		if self.neutrulesregex.exists(wd) :
			return -1
		return None

	"""
	wd : a word
	output : the list of the possible masculine inflections for this word
	"""
	def masc_inf(self, wd: str) :
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
	def detect_fem(self, wd : str):
		exc = ["France", "france"]
		masc = self.masc_inf(wd)
		if len(wd) < 6 or masc is None or wd in exc :
			return None
		else:
			if not self.dictrules.exists( wd.lower() ) and sum(1 if self.dictrules.exists( m.lower() ) else 0 for m in masc ) > 0 :
				return -1
			else:
				return None

	"""
	tokens : a list of Tokens (from a spacy model)
	output : a boolean indicating whether tks contains a coordination process
	"""
	def detect_coord(self, tks, start, end):
		for i in range(start, end):
			for j in range(i+1, end):
				itk = tks[i]
				jtk = tks[j]
				remain = [tks[k] for k in range(start, end) if k not in [i,j] ]
				if 'CCONJ' in [item.pos_ for item in remain] or 'PUNCT' in [item.pos_ for item in remain]:
					tA = self.unicoordrulesregex.match(itk.text + jtk.text)
					if tA is not None:
						k = min(max(0, i - (j-i-1-1)),i)
						ktk = tks[k]
						return Ann(ktk.idx, jtk.idx + len(jtk.text), text=tks[ktk.idx : jtk.idx + len(jtk.text)] ,metadata={"category" : ["coo"]} )
					tB = self.unicoordrulesregex.match(jtk.text + itk.text)
					if tB is not None:
						k = min(max(0, i - (j-i-1-1)),i)
						ktk = tks[k]
						return Ann(ktk.idx, jtk.idx + len(jtk.text), text=tks[ktk.idx : jtk.idx + len(jtk.text)] , metadata={"category" : ["coo"]} )
		return None

	"""
	doc : a spacy Document
	coord_range : the number of consecutive words to check in doc when detecting coordinations
	proc : the list of processes to detect in documents among the following : 'fle' for inflections, 'neu' for neutral-gender, 'fem' for feminisation, 'epi' for epicene words
	output : a list of Ann (annotations in the document)
	"""
	def detect_inc(self, doc, coord_range=4, proc=['fle', 'neu', 'fem', 'epi', 'coo']):
		res = []
		if 'coo' in proc :
			ind_grps = group(list( range(0, len(doc)) ), coord_range)
			for gr in ind_grps:
				tmp = self.detect_coord(doc, gr[0], gr[-1] + 1)
				if tmp is not None :
					res.append([tmp])
		for tk in doc :
			if 'fle' in proc :
				tmp = self.detect_flex(tk.text)
				if tmp is not None:
					res.append([Ann(tk.idx, tk.idx + len(tk.text), text=tk.text, metadata={"category" : ["fle"]}) ])
			if 'neu' in proc :
				tmp = self.detect_neut(tk.text)
				if tmp is not None:
					res.append([Ann(tk.idx, tk.idx + len(tk.text), text=tk.text, metadata={"category" : ["neu"]}) ])
			if 'fem' in proc :
				tmp = self.detect_fem(tk.text)
				if tmp is not None:
					res.append([Ann(tk.idx, tk.idx + len(tk.text), text=tk.text, metadata={"category" : ["fem"]}) ])
			if 'epi' in proc :
				tmp = self.detect_epi(tk.text)
				if tmp is not None:
					res.append([Ann(tk.idx, tk.idx + len(tk.text), text=tk.text, metadata={"category" : ["epi"]}) ])
		return merge_sort(res)

	def fit(self, x : list[str], y : list[list[Ann]]) -> None :
		print("This model doesn't need any kind of training : the 'fit' function is not doing anything")

	def _predict(self, x : list[str]) -> list[list[Ann]] :
		return [ self.detect_inc(self.nlp_model(item), proc=self.proc) for item in x ]
