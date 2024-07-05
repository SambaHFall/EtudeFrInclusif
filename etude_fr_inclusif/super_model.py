from ._utils import AnnPredModel, Ann
import pandas as pd
from etude_fr_inclusif.adv_rule_based import AdvRBModel
import re

"""

extends AnnPredModel
"""


class SuperAnnModel(AnnPredModel):

    def __init__(self, models=None, weights=None, tol=0.5):
        super().__init__(nlpmodel="fr_core_news_sm")
        if models is None:
            models = [AdvRBModel()]
        self.models = models
        if weights is None:
            self.weights = [1] * len(models)
        else:
            self.weights = weights
        self.tol = tol

    def fit(self, x: list[str], y: list[list[Ann]]) -> None:
        for mod in self.models:
            mod.fit(x, y)

    def _predict(self, x: list[str]) -> list[list[Ann]]:
        res = []

        for text in x:

            res.append([])
            pred_dfs = []

            for mod in self.models:
                pred_dfs.append(self._sub_shaping(text, mod.predict([text])[0]))

            super_dict = {"texts": [], "tokens": [], "annotated": [], "labels": [], "categories": []}

            for i in range(0, len(pred_dfs[0])):
                super_dict["texts"].append(pred_dfs[0]["texts"][i])
                super_dict["tokens"].append(pred_dfs[0]["tokens"][i])
                score = 0
                cat = []
                for j in range(0, len(pred_dfs)):
                    score += (1 if pred_dfs[j]["annotated"][i] else 0) * self.weights[j]
                    cat = cat + [item + f"_{j}" for item in pred_dfs[j]["categories"][i]]
                score = score / sum(w for w in self.weights)
                super_dict["annotated"].append( score >= 1 - self.tol)
                super_dict["labels"].append('I' if super_dict["annotated"] else 'O')
                super_dict["categories"].append(None if not super_dict["annotated"] else list(set(cat)))

            super_df = pd.DataFrame(super_dict)

            inproc = False
            cat = None
            for k, pred in super_df.iterrows():
                if (not inproc) and pred["annotated"]:
                    beg = pred["tokens"].idx
                    cat = pred["categories"]
                elif inproc and pred["annotated"]:
                    if any( re.match('coo.*', item) for item in cat) and any(re.match('coo.*', item) for item in pred["categories"]):
                        cat = cat + pred["categories"]
                    else:
                        last = super_df["tokens"][k - 1]
                        end = last.idx + len(last.text)
                        res[-1].append(Ann(beg, end, text=text[beg:end], metadata={"category": list(set(cat))}))
                        beg = pred["tokens"].idx
                        cat = pred["categories"]
                elif inproc and not pred["annotated"]:
                    last = super_df["tokens"][k - 1]
                    end = last.idx + len(last.text)
                    res[-1].append(Ann(beg, end, text=text[beg:end], metadata={"category": list(set(cat))}))
                    cat = None
                inproc = pred["annotated"]
            if inproc:
                res[-1].append(Ann(beg, len(text), text=text[beg:], metadata={"category": list(set(cat))}))

        return res


