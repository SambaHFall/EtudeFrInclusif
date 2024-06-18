from etude_fr_inclusif._utils import spacynlp
from etude_fr_inclusif.naive_rule_based import NaiveRBModel
import os
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# ajouter infos dans un petit doc (nb de phrases, nb de docs de Oscar, nb d'annotations, nb d'annotations par procédé)


def build_dataset(input_filename, out_filename="new_corpus.csv", loadspacyfrom="fr_core_news_sm") :

	model = NaiveRBModel()

	input_df = pd.read_csv("dataset/" + input_filename)

	if os.path.exists("dataset/" + out_filename) :
		df = pd.read_csv("dataset/" + out_filename)
	else :
		print("No previous file found")
		df = pd.DataFrame({"sample_id" : [], "text" : [], "tokens" : [], "spans" : [], "labels" : []})
	

	for i, content in tqdm(input_df.iterrows()) :
		if content["annotation"] :
			iddoc = content["iddoc"]
			text = content["context"]
			doc = model.nlp_model(text)
			row = df.loc[ df['sample_id'] == iddoc ]

			if len(row) == 0 :
				df = pd.concat([df,  pd.DataFrame({"sample_id" : [iddoc], "text" : [text], "tokens" : [ [tk.text for tk in doc] ], "spans" : [ [] ], "labels" : [ [] ]  }  ) ])
				row = df.loc[ df['sample_id'] == iddoc ]
			elif len(row) > 1 :
				print(len(row))
				raise Exception 				
			
			row = row.head().squeeze()

			rg = eval(content["range"])
			sp = (rg['beg'], rg['end']) 
			if str(sp) not in row["spans"] :
				row["spans"].append(sp)
				cat = content["category"]
				try:
					tab = eval(cat)
					cat = [item.split('_')[0] for item in tab]
					cat = list(set(cat))
				except Exception as e:
					pass

				row["labels"].append(cat)



	
	df.to_csv('dataset/' + out_filename, mode='w', index=False)


build_dataset("exploration_ann.csv")