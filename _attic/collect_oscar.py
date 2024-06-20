import sys
import os

script_dir = os.path.dirname( os.path.abspath(__file__) )
sur_script_dir = os.path.dirname(script_dir)
sys.path.append(sur_script_dir)

from etude_fr_inclusif.naive_rule_based import NaiveRBModel
from etude_fr_inclusif.crf import CRFModel
from etude_fr_inclusif.inclure import InclureModel
from etude_fr_inclusif.adv_rule_based import AdvRBModel
from etude_fr_inclusif.super_model import SuperAnnModel
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm

def explore_from_oscar(model, out_filename) :
	
	oscardata = load_dataset("oscar-corpus/OSCAR-2201", use_auth_token=True, language="fr", streaming=True, split="train")

	prog = 0
	if os.path.exists(sur_script_dir + '/dataset/progress.txt'):
		with open(sur_script_dir + '/dataset/progress.txt') as f:
			prog = int(f.read().strip())
		print(f'Skipping {prog} documents')
		oscardata = oscardata.skip(prog)
	
	if os.path.exists(sur_script_dir + '/dataset/docs_ref.csv') :
		df_refs = pd.read_csv(sur_script_dir + '/dataset/docs_ref.csv')
	else :
		df_refs = pd.DataFrame({"id" : [], "url" : [], "path" : []})

	if os.path.exists(sur_script_dir + "/dataset/" + out_filename) :
		df = pd.read_csv(sur_script_dir + "/dataset/" + out_filename)
	else :
		df = pd.DataFrame({"iddoc" : [], "context" : [], "range" : [], "text" : [], "category" : []})
	

	for i, d in enumerate(tqdm(oscardata)):
		try :
			if len(d['text']) < 1000000 :
				doc = model.nlp_model(d['text'])
				iddoc = d['id']
				texts = [s.text for s in doc.sents]
				preds = model.predict(texts)
				for j in range(0, len(texts)) :
					if len(preds[j]) > 0 :
						df = pd.concat([df, model.annotation_layout([texts[j]], [preds[j]], iddoc=[str(iddoc) + '_' + str(j) ], offset=None) ], ignore_index=True)
						if len(df_refs.loc[ df_refs['id'] == iddoc ]) == 0 :
							path = sur_script_dir + "/dataset/files/doc_" + str(iddoc) + ".txt"
							df_refs = pd.concat([df_refs, pd.DataFrame({"id" : [iddoc], "url" : [d['meta']['warc_headers']['warc-target-uri']], "path" : [path] })])
							with open(path, 'w') as f :
								f.write(d['text'])
						
		except KeyboardInterrupt :
			df.to_csv(sur_script_dir + '/dataset/' + out_filename, mode='w', index=False)
			df_refs.to_csv(sur_script_dir + '/dataset/docs_ref.csv', mode='w', index=False)
			with open(sur_script_dir + '/dataset/progress.txt', 'w') as f:
				f.write(f'{i + prog}')
			raise KeyboardInterrupt
		except :
			parts = out_filename.split('.')
			if len(parts) > 1 :
				err_out = '.'.join( parts[:-1] ) + '_err.' + parts[-1]
			else :
				err_out = out
			df.to_csv(sur_script_dir + '/dataset/' + err_out, mode='w', index=False)
			df_refs.to_csv(sur_script_dir + '/dataset/docs_ref.csv', mode='w', index=False)
			raise


args = sys.argv

if len(args) < 2 :
	model = SuperAnnModel(models=[AdvRBModel(proc=['fle', 'coo', 'neu', 'fem']), InclureModel()], weights=[0.5, 0.5], tol=0.6)
else :
	model = eval(args[1])


explore_from_oscar( 
	model , 
	'exploration.csv'
	)


