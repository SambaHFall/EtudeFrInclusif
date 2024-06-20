import sys
import os

script_dir = os.path.dirname( os.path.abspath(__file__) )
sur_script_dir = os.path.dirname(script_dir)
sys.path.append(sur_script_dir)

from etude_fr_inclusif._utils import spacynlp
from etude_fr_inclusif.naive_rule_based import NaiveRBModel

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import re

def compute_info_corpus(input_filename="new_corpus.csv", refs_filename="docs_ref.csv" ,out_filename="info_corpus.txt"):
	df = pd.read_csv(sur_script_dir + "/dataset/" + input_filename)

	refs = pd.read_csv(sur_script_dir + "/dataset/" + refs_filename)

	with open(sur_script_dir + "/dataset/" + out_filename, "w") as f :
		f.write( f"Nombre d'entrées : { len(df) }\n" )

		f.write("----------\n")
		docs_ids = list(set([ item.split('_')[0] for item in df["sample_id"] ]))
		f.write(f"Nombre de documents : {len(docs_ids)}\n") 
		f.write(f"Proportion par rapport au nombre de documents visités : { round(len(docs_ids) * 100 / int(max(docs_ids)),2) } %\n")

		ext_dict = {}
		for ind, content in refs.iterrows() :
			ext = re.sub( 'https?://', '', content["url"]).split('/')[0].split('.')[-1]
			if ext not in ext_dict:
				ext_dict[ext] = 0
			ext_dict[ext] += 1

		total_ext = sum( ext_dict[ext] for ext in ext_dict )

		ext_dict = sorted(ext_dict.items(), key=lambda x : x[1], reverse=True)

		ext_dict = [item for item in ext_dict if item[1] * 100 / total_ext > 1 ]

		other_ext = total_ext - sum( item[1] for item in ext_dict )

		for item in ext_dict :
			f.write(f"Nombre de document avec l'extension {item[0]} : {item[1]}  ({round(item[1] * 100 / total_ext,2)} %)\n")	
		f.write(f"Nombre de document avec une autre extension : {other_ext}  ({round(other_ext * 100 / total_ext,2)} %)\n")	

		f.write("----------\n")

		nb_anns = 0
		nb_anns_per_cat = {}
		for ind, content in df.iterrows() :
			labels = eval(content["labels"])
			nb_anns += len( labels )
			for it in labels :
				for lb in it :
					if lb not in nb_anns_per_cat :
						nb_anns_per_cat[lb] = 0
					nb_anns_per_cat[lb] += 1
		f.write(f"Nombre d'annotations : {nb_anns}\n")
		for lb in nb_anns_per_cat :
			f.write(f"Nombre d'annotations de catégorie {lb} : {nb_anns_per_cat[lb]}  ({round(nb_anns_per_cat[lb] * 100 / nb_anns,2)} %)\n")







def build_dataset(input_filename, out_filename="new_corpus.csv", loadspacyfrom="fr_core_news_sm") :

	model = NaiveRBModel()

	input_df = pd.read_csv(sur_script_dir + "/dataset/" + input_filename)

	if os.path.exists(sur_script_dir + "/dataset/" + out_filename) :
		df = pd.read_csv(sur_script_dir + "/dataset/" + out_filename)
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



	
	df.to_csv(sur_script_dir + '/dataset/' + out_filename, mode='w', index=False)


build_dataset("exploration_ann.csv")

compute_info_corpus()