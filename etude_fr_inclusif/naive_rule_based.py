from ._utils import AnnPredModel, Trie, Ann, group, merge_sort
import os
import Levenshtein
from unidecode import unidecode
import re

"""
class of annotation naive rule-based model
extends AnnPredModel
"""


class NaiveRBModel(AnnPredModel):

    def __init__(self):
        super().__init__()
        with open(os.path.dirname(os.path.abspath(__file__)) + '/data/dictrules.txt') as f:
            self.fr_dict = Trie(f.read().split('\n'))

    """
    tks : a list of Tokens (from a spacy model)
    prox : the maximum distance that is allowed for two words to be considered as close
    output : a boolean indicating whether tks contains a coordination process
    """

    def coord(self, tks, prox):
        for first_i in range(0, len(tks)):
            first = tks[first_i]
            for last_i in range(first_i + 1, len(tks)):
                last = tks[last_i]
                remain_pos = [tks[k].pos_ for k in range(0, len(tks)) if k not in [first_i, last_i] ]
                if first.lemma_ == last.lemma_ and 0 < Levenshtein.distance(first.text, last.text) <= prox and ('CCONJ' in remain_pos or 'PUNCT' in remain_pos):
                    return Ann(first.idx, last.idx + len(last.text), metadata={"category": ["coo"]}), last_i
        return None, None

    """
    doc : a spacy Document
    n : the number of consecutive words to check in doc when detecting coordinations
    prox : the maximum distance between two words for them to be considered as close
    output : a list of Ann (all the annotations corresponding to coordinations found in the document)
    """

    def detect_coord(self, doc, n=4, prox=3) -> list[Ann]:
        ngrams = group([tk for tk in doc], n)
        anns = []
        i = 0
        while i < len(ngrams):
            gr = ngrams[i]
            ann, offset = self.coord(gr, prox)
            if ann is not None:
                ann.text = doc.text[gr[0].idx: gr[offset].idx + len(gr[offset].text)]
                anns.append(ann)
                i += offset
            i += 1
        return anns

    """
    doc : a spacy Document
    output : a list of Ann (all the annotations corresponding to feminisations found in the document)
    """

    def detect_fem(self, doc) -> list[Ann]:
        femregex = re.compile(r'.*(ère|ice|eure|elle|effe|ette|esse|enne|euse)s?$')
        anns = []
        for tk in doc:
            wd = tk.text
            raw = unidecode(wd).lower()
            if femregex.match(wd) and not self.fr_dict.exists(raw):
                anns.append(Ann(tk.idx, tk.idx + len(tk.text), metadata={"category": ["fem"]}, text=wd))
        return anns

    """
    wd : astring
    output : the index of the last character of the word which is neither a lower case character nor a numeral (or None if no only lower case characters and numerals found)
    """

    def last_non_alpha_char(self, wd: str):
        cpt = 0
        last = None
        for char in wd:
            cpt += 1
            asc = ord(char)
            if not(asc >= 97 and asc <= 122) and not (48 <= asc <= 57):
                last = cpt
        return last

    """
    doc : a spacy Document
    output : a list of Ann (all the annotations corresponding to inflections found in the document)
    """

    def detect_flex(self, doc) -> list[Ann]:
        anns = []
        for tk in doc:
            last = self.last_non_alpha_char(unidecode(tk.text))
            if last is not None and (last >= len(tk.text) / 2) and (len(tk.text) - last < 4 ) and len(tk.text) > 4:
                anns.append(Ann(tk.idx, tk.idx + len(tk.text), metadata={"category": ["fle"]}, text=tk.text))
        return anns

    """
    doc : a spacy Document
    output : a list of Ann (all the annotations corresponding to neutralisations found in the document)
    """

    def detect_neut(self, doc) -> list[Ann]:
        neutregex = re.compile(r'.*æ.*')
        anns = []
        for tk in doc:
            wd = tk.text
            raw = unidecode(wd).lower()
            if neutregex.match(wd) and not self.fr_dict.exists(raw):
                anns.append(Ann(tk.idx, tk.idx + len(tk.text), metadata={"category": ["neu"]}, text=wd))
        return anns

    """
    doc : a spacy Document
    output : a list of Ann (all the annotations corresponding to inclusive french processes found in the document)
    """

    def detect_inc(self, doc) -> list[Ann]:
        flex_anns = self.detect_flex(doc)
        fem_anns = self.detect_fem(doc)
        neut_anns = self.detect_neut(doc)
        coord_anns = self.detect_coord(doc)
        return merge_sort([flex_anns, fem_anns, neut_anns, coord_anns])

    def fit(self, x: list[str], y: list[list[Ann]]) -> None:
        print("This model doesn't need any kind of training : the 'fit' function is not doing anything")

    def _predict(self, x: list[str]) -> list[list[Ann]]:
        return [self.detect_inc(self.nlp_model(item)) for item in x]
