import sys
import os

script_dir = os.path.dirname( os.path.abspath(__file__) )
sur_script_dir = os.path.dirname(script_dir)
sys.path.append(sur_script_dir)

import pandas as pd
from etude_fr_inclusif import NaiveRBModel
import matplotlib.pyplot as plt
from datasets import load_dataset


"""
Inspect the final corpus file to compute the distribution curve of the length (= number of token OR number of characters) of the documents, 
and compare it to a same-sized sample of random documents from OSCAR
"""

def histo(data_bias, data_ref, output_filename="repartition_data.png", neglig=5) :
	x_bias = sorted(list(set(data_bias) ) )
	x_ref = sorted(list(set(data_ref) ) )

	y_bias = [0] * len(x_bias)
	y_ref = [0] * len(x_ref)

	for item in data_bias :
		y_bias[ x_bias.index(item) ] += 1

	for item in data_ref :
		y_ref[ x_ref.index(item) ] += 1

	max_bias = max(i + 1 for i in range(0, len(y_bias)) if y_bias[i] > neglig )	
	max_ref = max(i + 1 for i in range(0, len(y_ref)) if y_ref[i] > neglig )	

	max_both = max(max_bias, max_ref)

	fig, ax = plt.subplots()
	ax.plot(x_bias[:max_both],y_bias[:max_both])
	ax.plot(x_ref[:max_both], y_ref[:max_both])
	fig.legend(['biaisé', 'ref'])
	plt.savefig(sur_script_dir + "/dataset/stats/" + output_filename)

def test_repartition(input_filename, test_char=False) :

	model = NaiveRBModel()

	input_df = pd.read_csv(sur_script_dir + "/dataset/" + input_filename)
	if test_char :
		biased_val = [ len(item) for item in input_df["text"] ]
	else :
		biased_val = [ len(eval(item)) for item in input_df["tokens"] ]
	n = len(biased_val)

	ref_val = []
	cpt = 0
	oscardata = load_dataset("oscar-corpus/OSCAR-2201", use_auth_token=True, language="fr", streaming=True, split="train")
	for d in oscardata :
		if len(d['text']) < 1000000 :
			cpt += 1
			doc = model.nlp_model(d['text'])
			for s in doc.sents :
				cpt += 1
				ref_val.append( len(s.text) if test_char else len(s) )
				if cpt >= n :
					break
			if cpt >= n :
					break
		if cpt >= n :
				break

	histo(biased_val, ref_val, output_filename="repartition_" + ("char" if test_char else "tokens") + "_biais_ref.png", neglig=8 if test_char else 4 )


test_repartition("new_corpus.csv")
test_repartition("new_corpus.csv", test_char=True)