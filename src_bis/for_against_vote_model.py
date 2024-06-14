from ._utils import *
import pandas as pd
from etude_fr_inclusif.inclure import InclureModel

"""

extends AnnPredModel
"""
class FAVoteModel(AnnPredModel) :

	def __init__(self, for_models=[InclureModel()], ag_models=[], weights=None, tol=0.5):
		super().__init__(nlpmodel="fr_core_news_sm")
		self.for_models = for_models
		self.ag_models = ag_models
		if weights is None :
			self.weights = [[1] * len(for_models), [1] * len(ag_models) ] 
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

			pred_for_dfs = []
			pred_ag_dfs = []

			for mod in self.for_models :
				pred_for_dfs.append( self._sub_shaping(text, mod.predict([text])[0] ) )

			for mod in self.ag_models :
				pred_ag_dfs.append( self._sub_shaping(text, mod.predict([text])[0] ) )

			super_dict = {"texts" : [], "tokens" : [], "annotated" : [], "labels" : [], "categories" : []}

			for i in range(0, len(pred_for_dfs[0]) ) :
				super_dict["texts"].append(pred_for_dfs[0]["texts"][i])
				super_dict["tokens"].append(pred_for_dfs[0]["tokens"][i])
				score = 0
				cat = []
				for j in range(0, len(pred_for_dfs)) :
					score += (1 if pred_for_dfs[j]["annotated"][i] else 0) * self.weights[0][j]
					cat = cat + [item + f"_{j}" for item in pred_dfs[j]["categories"][i] ]
				for j in range(0, len(pred_ag_dfs)) :
					score += (-1 if pred_ag_dfs[j]["annotated"][i] else 0) * self.weights[1][j]
				score = score / ( sum(w for w in self.weights[0]) + sum(w for w in self.weights[1]) )
				super_dict["annotated"].append( score >= 1 - self.tol )
				super_dict["labels"].append('I' if super_dict["annotated"] else 'O')
				super_dict["categories"].append(None if not super_dict["annotated"] else list(set(cat)) )

			super_df = pd.DataFrame(super_dict)

			inproc = False
			cat = None
			for k, pred in super_df.iterrows() :
				if (not inproc) and pred["annotated"] :
					beg = pred["tokens"].idx
					cat = pred["categories"]
				elif inproc and pred["annotated"] :
					cat = cat + pred["categories"]
				elif inproc and not pred["annotated"]  :
					last = super_df["tokens"][k-1]
					end = last.idx + len(last.text) 
					res[-1].append(Ann(beg, end, text=text[beg:end], metadata={"category" : list(set(cat))}) )
					cat = None
				inproc = pred["annotated"]
			if inproc :
				res[-1].append(Ann(beg, len(text), text=text[beg:], metadata={"category" : list(set(cat))}))

		return res


