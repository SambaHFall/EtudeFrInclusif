from src._utils import *
from src.naive_rule_based import NaiveRBModel
from src.crf import CRFModel
from src.inclure import InclureModel
from src.adv_rule_based import AdvRBModel
from src.adv_rule_based import *
from src.super_model import SuperAnnModel
import pandas as pd

import os

from datasets import load_dataset
from tqdm import tqdm


oscardata = load_dataset("oscar-corpus/OSCAR-2201", use_auth_token=True, language="fr", streaming=True, split="train")

def load_new_dataset_from_oscar() :
	prog = 0
	if os.path.exists('progress.txt'):
		with open('progress.txt') as f:
			prog = int(f.read().strip())
		print(f'Skipping {prog} documents')
		for i in tqdm(range(prog)):
			next(oscardata)

	modelSUPER = SuperAnnModel(models=[AdvRBModel(proc=['fle', 'coo']) , NaiveRBModel(), InclureModel(), CRFModel()], weights=[0.29, 0.5, 0.5, 0.26], tol=0.6)

	x, y = get_fr_inclusif_data()

	modelSUPER.models[-1].fit(x,y)

	df = pd.DataFrame({"iddoc" : [], "context" : [], "range" : [], "text" : [], "category" : [] })

	for i, d in enumerate(tqdm(oscardata)):
		try :
			if len(d['text']) < 1000000 :
				iddoc = d['id']
				preds = modelSUPER.predict([d['text']])
				if len(preds[0]) > 0 :
					df = pd.concat([df, modelSUPER.annotation_layout([d['text']], preds, iddoc=[iddoc]) ], ignore_index=True)
		except KeyboardInterrupt :
			with open('progress.txt', 'w') as f:
				df.to_csv("dataset/SUPER_new_dataset.csv", mode='a')
				f.write(f'{i + prog}')
			raise KeyboardInterrupt


load_new_dataset_from_oscar()



