from ._utils import AnnPredModel, merge_sort, Ann
from etude_fr_inclusif.code_inclure.x import domain_and_exts, different_gender, sub

"""
class of annotation INCLURE-method-based model
extends AnnPredModel
"""


class InclureModel(AnnPredModel):

    def __init__(self, proc=['fle', 'coo']):
        super().__init__(nlpmodel="fr_dep_news_trf")
        self.proc = proc

    """
    doc : a spacy Document
    proc : list of french inclusive categories to detect
    output : a list of annotations found in that document (using a method really close to the one used in INCLURE)
    """
    def detect_inc(self,doc, proc) -> list[Ann]:
        if domain_and_exts.search(doc.text) is not None:
            return []
        res = []
        for token in doc:
            if token.like_url or token.like_email:
                return []
            if token.dep_ == "ROOT" or token.pos_ in {"VERB", "AUX"} or token.is_space:
                continue
            if 'coo' in proc:
                if token.lemma == token.head.lemma and different_gender(token, token.head):
                    ind_i = token.i
                    ind_j = token.head.i
                    if ind_j < ind_i:
                        ind_i, ind_j = ind_j, ind_i
                    k = min(max(0, ind_i - (ind_j - ind_i - 1 - 1)), ind_i)
                    beg = doc[k].idx
                    end = doc[ind_j].idx + len(doc[ind_j])

                    res.append([Ann(beg, end, text=doc.text[beg: end], metadata={"category": ["coo"]})])
            if 'fle' in proc:
                x_text = sub(token.text)
                if x_text is not None and x_text != token.text:
                    res.append([Ann(token.idx, token.idx + len(token.text), text=token.text, metadata={"category": ["fle"]})])
        return merge_sort(res)

    def fit(self, x: list[str], y: list[list[Ann]]) -> None:
        print("This model doesn't need any kind of training : the 'fit' function is not doing anything")

    def _predict(self, x: list[str]) -> list[list[Ann]]:
        return [ self.detect_inc(self.nlp_model(item), proc=self.proc) for item in x]
