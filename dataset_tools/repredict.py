import sys
import os

script_dir = os.path.dirname( os.path.abspath(__file__) )
sur_script_dir = os.path.dirname(script_dir)
sys.path.append(sur_script_dir)

import pandas as pd
from etude_fr_inclusif.adv_rule_based import AdvRBModel
from etude_fr_inclusif.inclure import InclureModel
from etude_fr_inclusif.super_model import SuperAnnModel
from tqdm import tqdm

"""
produce a 'corrected_exploration.csv' from the original 'exploration.csv' file by running again inclusive french detection models on some particular documents
"""

def repredict_zone(filename, model, start=0, end=None) :

	df = pd.read_csv(sur_script_dir + '/dataset/' + filename)
	
	if end is None :
		end = len(df)

	res_df = {key : [] for key in df.columns}

	for i, content in df[start:end].iterrows() :
		text = content['context']
		text_id = content['iddoc']
		anns = model.predict(text)
		for ann in anns :
			res_df['iddoc'].append(text_id)
			res_df["context"].append(text)
			res_df["range"].append( {'beg' : ann.beg, 'end' : ann.end} )
			res_df["text"].append(ann.text)
			res_df["category"].append(ann.metadata["category"])
			res_df["annotation"].append('')

	prec_df = df[:start]
	succ_df = df[end:]
	res_df = pd.DataFrame(res_df)

	res_df = pd.concat([prec_df, res_df, succ_df])

	res_df.to_csv(sur_script_dir + '/dataset/' + filename, mode='w', index=False)


def repredict_all_trues(filename, model, start=0, end=None) :

	df = pd.read_csv(sur_script_dir + '/dataset/' + filename)
	
	if end is None :
		end = len(df)

	res_df = {key : [] for key in df.columns}

	ids = []

	for i, content in tqdm(df[start:end].iterrows()) :
		if not content["annotation"] :
			for key in df.columns :
				res_df[key].append(content[key])
		else :
			text = content['context']
			text_id = content['iddoc']
			if text_id not in ids :
				ids.append(text_id)
				anns = model.predict(text)
				for ann in anns :
					res_df['iddoc'].append(text_id)
					res_df["context"].append(text)
					res_df["range"].append( {'beg' : ann.beg, 'end' : ann.end} )
					res_df["text"].append(ann.text)
					res_df["category"].append(ann.metadata["category"])
					res_df["annotation"].append('')

	pd.DataFrame(res_df).to_csv(sur_script_dir + '/dataset/corrected_' + filename, mode='w', index=False)


repredict_all_trues('exploration_ann.csv', SuperAnnModel(models=[AdvRBModel(proc=['fle', 'coo', 'neu', 'fem']), InclureModel()], weights=[0.5, 0.5], tol=0.8) )



	




