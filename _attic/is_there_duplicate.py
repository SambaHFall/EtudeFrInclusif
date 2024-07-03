import os

script_dir = os.path.dirname( os.path.abspath(__file__) )
sur_script_dir = os.path.dirname(script_dir)

import pandas as pd

input_df = pd.read_csv(sur_script_dir + "/dataset/new_corpus.csv")

duplics = []

for ind, content in input_df.iterrows() :
	spans = eval(content["spans"])
	dup = False
	for i in range(0, len(spans)) :
		for j in range(i+1, len(spans)) :
			if spans[i][0] == spans[j][0] and spans[i][1] == spans[j][1] :
				dup = True
	if dup :
		duplics.append(ind)

print(f'Duplicat ({len(duplics)}): {duplics}')
