from ._utils import *
import pandas as pd
from src.inclure import InclureModel

"""

extends AnnPredModel
"""
class SuperAnnModel(AnnPredModel) :

	def __init__(self, models=[InclureModel()], weights=None, tol=0.5):
		super().__init__(nlpmodel="fr_core_news_sm")
		self.models = models
		if weights is None :
			self.weights = [1] * len(models)
		else:
			self.weights = weights
		self.tol = tol

	def fit(self, x : list[str], y : list[list[Ann]]) -> None :
		for mod in self.models :
			mod.fit(x,y)

	def predict(self, x : list[str]) -> list[list[Ann]] :
		
		res = []

		for text in x :

			res.append([])

			pred_dfs = []

			for mod in self.models :
				pred_dfs.append( self._sub_shaping(text, mod.predict([text])[0] ) )

			super_dict = {"texts" : [], "tokens" : [], "annotated" : [], "labels" : []}

			for i in range(0, len(pred_dfs[0]) ) :
				super_dict["texts"].append(pred_dfs[0]["texts"][i])
				super_dict["tokens"].append(pred_dfs[0]["tokens"][i])
				score = 0
				for j in range(0, len(pred_dfs)) :
					score += (1 if pred_dfs[j]["annotated"][i] else 0) * self.weights[j]
				score = score / sum(w for w in self.weights)
				super_dict["annotated"].append( score >= 1 - self.tol )
				super_dict["labels"].append('I' if super_dict["annotated"] else 'O')

			super_df = pd.DataFrame(super_dict)

			inproc = False
			for k, pred in super_df.iterrows() :
				if (not inproc) and pred["annotated"] :
					beg = pred["tokens"].idx
				elif inproc and not pred["annotated"]  :
					last = super_df["tokens"][k-1]
					end = last.idx + len(last.text) 
					res[-1].append(Ann(beg, end, text=text[beg:end] ) )
				inproc = pred["annotated"]
			if inproc :
				res[-1].append(Ann(beg, len(text), text=text[beg:] ))

		return res


