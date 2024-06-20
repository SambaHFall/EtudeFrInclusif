import sys
import os

script_dir = os.path.dirname( os.path.abspath(__file__) )
sur_script_dir = os.path.dirname(script_dir)

from datasets import load_dataset
import csv
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer


kwargs = {
    "homepage": "https://github.com/grouin/corpus-francais-inclusif",
    "citation": """@inproceedings{grouin-2022-impact,
        title = "Impact du fran{\c{c}}ais inclusif sur les outils du {TAL} (Impact of {F}rench Inclusive Language on {NLP} Tools)",
        author = "Grouin, Cyril",
        editor = "Est{\`e}ve, Yannick  and
          Jim{\'e}nez, Tania  and
          Parcollet, Titouan  and
          Zanon Boito, Marcely",
        booktitle = "Actes de la 29e Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles. Volume 1 : conf{\'e}rence principale",
        month = "6",
        year = "2022",
        address = "Avignon, France",
        publisher = "ATALA",
        url = "https://aclanthology.org/2022.jeptalnrecital-taln.12",
        pages = "126--135",
        abstract = "Le fran{\c{c}}ais inclusif est une vari{\'e}t{\'e} du fran{\c{c}}ais standard mise en avant pour t{\'e}moigner d{'}une conscience de genre et d{'}identit{\'e}. Plusieurs proc{\'e}d{\'e}s existent pour lutter contre l{'}utilisation g{\'e}n{\'e}rique du masculin (coordination de formes f{\'e}minines et masculines, f{\'e}minisation des fonctions, {\'e}criture inclusive, et neutralisation). Dans cette {\'e}tude, nous nous int{\'e}ressons aux performances des outils sur quelques t{\^a}ches du TAL ({\'e}tiquetage, lemmatisation, rep{\'e}rage d{'}entit{\'e}s nomm{\'e}es) appliqu{\'e}s sur des productions langagi{\`e}res de ce type. Les taux d{'}erreur sur l{'}{\'e}tiquetage en parties du discours (TreeTagger et spaCy) augmentent de 3 {\`a} 7 points sur les portions r{\'e}dig{\'e}es en fran{\c{c}}ais inclusif par rapport au fran{\c{c}}ais standard, sans lemmatisation possible pour le TreeTagger. Sur le rep{\'e}rage d{'}entit{\'e}s nomm{\'e}es, les mod{\`e}les sont sensibles aux contextes en fran{\c{c}}ais inclusif et font des pr{\'e}dictions erron{\'e}es, avec une pr{\'e}cision en baisse.",
        language = "French",
    }""",
    # Nom des 3 dossiers
    "split_paths": {
        "multi": "multi",
        "vfi": "vfi",
        "vfs": "vfs"
    }
}
# 'dfki-nlp/brat' correspond au code qui permet de charger les annotations au format BRAT
dataset = load_dataset('dfki-nlp/brat', data_dir="corpus-francais-inclusif", **kwargs)

header = ["Text", "Annotations"]
content = []

for item in dataset['vfi'] :
    text = item['context']
    sent_ranges = PunktSentenceTokenizer().span_tokenize(text)
    sent_ranges = [item for item in sent_ranges]
    tmp = []
    for rg in sent_ranges :
        tmp.append([ text[rg[0]:rg[1]], [] ])

    sp = item['spans']
    att = item['attributions']
    for i in range(0, len(sp['id']) ):
        if sp['type'][i] == 'Inclusif' :
            target = sp['id'][i]
            rg = sp['locations'][i]
            cpt = 0
            while rg['start'][0] > sent_ranges[cpt][1]:
                cpt += 1
            if rg['end'][0] > sent_ranges[cpt][1] :
                print("Erreur : chevauchement dans le découpage")
                continue
            cat = None
            for j in range(0,len(att['target'])) :
                if att['target'][j] == target:
                    cat = att['value'][j][:3]
            tmp[cpt][1].append({ "beg" : rg['start'][0] - sent_ranges[cpt][0], "end" : rg['end'][0] - sent_ranges[cpt][0], "text" : sp['text'][i], "metadata" : ({"category" : [cat]} if cat is not None else {}) } )
    content = content + tmp

with open(sur_script_dir + "/data/fr_inclusif.csv", 'w') as f :
    csvwriter = csv.writer(f)
    csvwriter.writerow(header)
    csvwriter.writerows(content)


