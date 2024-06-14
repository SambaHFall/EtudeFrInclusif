from etude_fr_inclusif._utils import *
from etude_fr_inclusif.naive_rule_based import NaiveRBModel
from etude_fr_inclusif.crf import CRFModel
from etude_fr_inclusif.inclure import InclureModel
from etude_fr_inclusif.adv_rule_based import AdvRBModel
from etude_fr_inclusif.adv_rule_based import *
from etude_fr_inclusif.super_model import SuperAnnModel
import pandas as pd

import os

from datasets import load_dataset
from tqdm import tqdm


def load_new_dataset_from_oscar(model, out) :
	
	oscardata = load_dataset("oscar-corpus/OSCAR-2201", use_auth_token=True, language="fr", streaming=True, split="train")

	prog = 0
	if os.path.exists('progress.txt'):
		with open('progress.txt') as f:
			prog = int(f.read().strip())
		print(f'Skipping {prog} documents')
		for i in tqdm(range(prog)):
			oscardata = oscardata.skip(1)
	
	if os.path.exists('dataset/docs_ref.csv') :
		df_refs = pd.read_csv('dataset/docs_ref.csv')
	else :
		df_refs = pd.DataFrame({"id" : [], "url" : []})

	if os.path.exists(out) :
		df = pd.read_csv(out)
	else :
		df = pd.DataFrame({"iddoc" : [], "context" : [], "range" : [], "text" : [], "category" : []})
	

	for i, d in enumerate(tqdm(oscardata)):
		try :
			if len(d['text']) < 1000000 :
				doc = model.nlp(d['text'])
				iddoc = d['id']
				texts = [s.text for s in doc.sents]
				preds = model.predict(texts)
				for j in range(0, len(texts)) :
					if len(preds[j]) > 0 :
						df = pd.concat([df, model.annotation_layout([texts[j]], [preds[j]], iddoc=[str(iddoc) + '_' + str(j) ], offset=None) ], ignore_index=True)
						if len(df_refs['id'].isin([iddoc])) == 0 :
							df_refs = pd.concat([df_refs, pd.DataFrame({"id" : [iddoc], "url" : [d['meta']['warc_headers']['warc-target-uri']]})])
						
		except KeyboardInterrupt :
			with open('progress.txt', 'w') as f:
				df.to_csv(out, mode='w', index=False)
				df_refs.to_csv('dataset/docs_ref.csv', mode='w', index=False)
				f.write(f'{i + prog}')
			raise KeyboardInterrupt
		except :
			parts = out.split('.')
			if len(parts) > 1 :
				err_out = '.'.join( parts[:-1] ) + '_err.' + parts[-1]
			else :
				err_out = out
			df.to_csv(err_out, mode='w', index=False)
			df_refs.to_csv('dataset/docs_ref.csv', mode='w', index=False)
			raise

load_new_dataset_from_oscar( 
	SuperAnnModel(models=[AdvRBModel(proc=['fle', 'coo', 'neu', 'fem']) , NaiveRBModel(), InclureModel()], weights=[0.29, 0.45, 0.26], tol=0.8), 
	'dataset/SUPER_new_dataset_v2.csv'
	)



