from transformers import pipeline
from ._utils import *

ner = pipeline('token-classification', model='CATIE-AQ/NERmembert-base-3entities', tokenizer='CATIE-AQ/NERmembert-base-3entities', aggregation_strategy="simple")


class NERModel(AnnPredModel) :

	"""
	proc : the list of processes to detect in documents among the following : 'fle' for inflections, 'neu' for neutral-gender, 'fem' for feminisation, 'epi' for epicene words
	"""
	def __init__(self):
		super().__init__()

	def fit(self, x : list[str], y : list[list[Ann]]) -> None :
		print("This model doesn't need any kind of training : the 'fit' function is not doing anything")

	def predict(self, x : list[str]) -> list[list[Ann]] :
		res = []
		for text in x :
			pred = ner(text)
			res.append([])
			for p in pred:
				res[-1].append(Ann(p['start'], p['end'], text=p['word']) )
		return res
				
