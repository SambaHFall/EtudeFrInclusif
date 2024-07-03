import os
import spacy
import json
import numpy as np
from tqdm import tqdm

script_path = os.path.dirname(os.path.realpath(__file__)) 
sur_script_path = os.path.dirname(script_path)
path = sur_script_path + '/dataset/files/'

"""
Produce a similar_files.json file, which gives for each document, all the documents that are similar
"""

nlp = spacy.load("fr_core_news_sm")

def levenshtein_for_list(list_a, list_b, threshold=None) :
	mat = np.zeros( (len(list_a) + 1, len(list_b) + 1) )

	for i in range(1, len(list_a)+1) :
		mat[i][0] = i

	for j in range(1, len(list_b)+1) :
		mat[0][j] = j


	mat[-1][-1] = None

	for i in range(1, len(list_a)+1) :
		for j in range(1, len(list_b) +1) :
			cost = 0 if list_a[i-1] == list_b[j-1] else 1
			mat[i][j] = min(mat[i-1][j] + 1, mat[i][j-1] + 1, mat[i-1][j-1] + cost )

		if np.max(mat[i]) > threshold :
			break
	return mat[-1][-1]


files = os.listdir(path)

raw_files = []

for file in files :
	with open(path+file) as f :
		raw_files.append(f.read().strip())

files = [ item[4:-4] for item in files  ]

files_dict = { item : [] for item in files }

with open(sur_script_path + '/same_files.txt', 'w') as f :
	for i in tqdm(range(0, len(raw_files))) :
		for j in tqdm(range(i+1, len(raw_files))) :
			doc_i = nlp(raw_files[i])
			doc_j = nlp(raw_files[j])
			i_threshold = 0.05 * len(doc_i)
			j_threshold = 0.05 * len(doc_j)
			dist = levenshtein_for_list( [item.text for item in doc_i] , [item.text for item in doc_j], threshold=max(i_threshold, j_threshold))
			if dist is not None and dist <= i_threshold :
				files_dict[ files[j] ].append( files[i] )
			if dist is not None and dist <= j_threshold:
				files_dict[ files[i] ].append( files[j] )
			


with open(script_path + '/similar_files.txt', 'w') as f :
	f.write( json.dumps(files_dict, indent=4) )