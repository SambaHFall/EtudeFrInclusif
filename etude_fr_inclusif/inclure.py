from ._utils import AnnPredModel, merge_sort, Ann
from inclure.x import domain_and_exts, different_gender, sub
import re

"""
doc : a spacy Document
output : a list of annotations found in that document (using a method really close to the one used in INCLURE)
"""
def detect_inc(doc) -> list[Ann] :
	if domain_and_exts.search(doc.text) is not None:
		return []
	res = []
	for token in doc:
		if token.like_url or token.like_email:
			return []
		if token.dep_=="ROOT" or token.pos_ in {"VERB","AUX"} or token.is_space:
			continue
		if token.lemma == token.head.lemma and different_gender(token, token.head):
			beg_i = token.idx
			end_i = token.idx + len(token.text)
			beg_j = token.head.idx
			end_j = token.head.idx + len(token.head.text)
			if beg_j < beg_i:
				beg_i, beg_j = beg_j, beg_i
				end_i, end_j = end_j, end_i
			res.append( [Ann(beg_i, end_j, text=doc.text[beg_i : end_j], metadata={"category" : ["coo"]})] )
		x_text = sub(token.text)
		if x_text is not None and x_text != token.text :
			res.append( [Ann(token.idx, token.idx + len(token.text), text=token.text, metadata={"category" : ["fle"]})] )
	return merge_sort(res)
    	
"""
class of annotation INCLURE-method-based model
extends AnnPredModel
"""
class InclureModel(AnnPredModel) :

	def __init__(self):
		super().__init__(nlpmodel="fr_dep_news_trf")

	def fit(self, x : list[str], y : list[list[Ann]]) -> None :
		print("This model doesn't need any kind of training : the 'fit' function is not doing anything")

	def predict(self, x : list[str]) -> list[list[Ann]] :
		return [ detect_inc(self.nlp_model(item)) for item in x ]
