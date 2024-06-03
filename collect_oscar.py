from src._utils import *
from src.naive_rule_based import NaiveRBModel
from src.crf import CRFModel
from src.inclure import InclureModel
from src.adv_rule_based import AdvRBModel
from src.adv_rule_based import *
from src.super_model import SuperAnnModel
import pandas as pd

from datasets import load_dataset


oscardataset = load_dataset("oscar-corpus/OSCAR-2201", use_auth_token=True, language="fr", streaming=True, split="train")

def load_new_dataset_from_oscar() :
	modelSUPER = SuperAnnModel(models=[AdvRBModel(proc=['fle', 'coo']) , NaiveRBModel(), InclureModel()], weights=[1, 0.46, 1], tol=0.6)
	cpt = 0
	cur = 0
	goal = 1000
	maxiter = 20000

	df = pd.DataFrame({"iddoc" : [], "context" : [], "range" : [], "text" : [], "category" : [] })

	for d in oscardataset :
		if len(d['text']) < 1000000 :
			iddoc = d['id']
			preds = modelSUPER.predict([d['text']])
			if len(preds[0]) > 0 :
				cpt += 1
				df = pd.concat([df, modelSUPER.annotation_layout([d['text']], preds, iddoc=[iddoc]) ], ignore_index=True)

			if cpt < goal and cur < maxiter:
				cur += 1
				if cur % 20 == 0 :
					with open("adv.text", 'w') as f :
						f.write("nbiter=" + str(cur) + " / nbdocs=" + str(cpt) )
					df.to_csv("dataset/SUPER_new_dataset.csv")
			else :
				df.to_csv("dataset/SUPER_new_dataset.csv")
				break

load_new_dataset_from_oscar()



