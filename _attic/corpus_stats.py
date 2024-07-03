import sys
import os

script_dir = os.path.dirname( os.path.abspath(__file__) )
sur_script_dir = os.path.dirname(script_dir)
sys.path.append(sur_script_dir)

import pandas as pd
import numpy as np
import re

"""
Inspect the corpus final file to compute statistics about the different kind of uses for each inclusive french category
Produces at most 6 csv files in dataset/stats :
- 5 files giving the number of occurrences for every form of one particular french inclusive category (1 file for each category, none for categories that have no occurrences in the corpus)
- 1 file giving the number of occurrences for every url extension depending on the categories of inclusive french expression that were found in the document
"""


def tokens_span(text, tokens, span=None) :
	if span is None :
		span = (0, len(text))
	res = {}
	ind = 0
	for tk in tokens :
		start = tk[0]
		while text[ind] != start :
			ind += 1
		if ind >= span[0] and ind + len(tk) <= span[1] :
			res[tk] = (ind, ind + len(tk))
		ind = ind + len(tk)
	return res

def compute_stats(input_filename="new_corpus.csv") :

	freq_dict = {'coo':{}, 'fle':{}, 'fem':{}, 'neu':{}, 'epi':{} }
	freq_df = {'coo': {'text' : [], 'occ' : []}, 'fle':{'end' : [], 'signs' : [], 'occ' : []}, 'fem':{'word' : [], 'occ' : []}, 'neu':{'word' : [], 'occ' : []}, 'epi':{'word' : [], 'occ' : []} }

	ext_dict = {'coo':{}, 'fle':{}, 'fem':{}, 'neu':{}, 'epi':{} }

	df = pd.read_csv(sur_script_dir + "/dataset/" + input_filename)
	docs_ref = pd.read_csv(sur_script_dir + "/dataset/docs_ref.csv")

	for ind, content in df.iterrows() :
		doc_id = float(content['sample_id'].split('_')[0])

		ext = (docs_ref.loc[docs_ref["id"] == doc_id]).squeeze()["url"].split("://")[1].split("/")[0].split('.')[-1]

		lbs = eval(content['labels'])
		sps = eval(content['spans'])
		for i in range(0, len(lbs)) :
			start = sps[i][0]
			end = sps[i][1]
			text = content['text'][start:end]
			tks_span = sorted( tokens_span(content['text'], eval(content['tokens']), span=(start, end)).items(), key = lambda x : x[1][0] )
			lb = lbs[i]


			if 'coo' in lb :
				btw = ''
				if len(tks_span) > 2 :
					btw = btw + content['text'][ tks_span[0][1][1]  : tks_span[-1][1][0] - 1 ]
					btw = btw.strip()
				if re.match( r'\s*$', btw) :
					btw = '_'
				if btw not in freq_dict['coo'] :
					freq_dict['coo'][btw] = 0
				freq_dict['coo'][btw] += 1
			if 'fle' in lb :
				punct = "[^a-zA-Z0-9çàâéèêëîïôöûüæœ]"
				for tk_span in tks_span :
					tk = tk_span[0]
					ind = 0
					while ind < len(tk) and not re.match( punct , tk[ind].lower()) :
						ind += 1
					if ind < len(tk) :
						suff = tk[ind:]
						if suff not in freq_dict['fle'] :
							freq_dict['fle'][suff] = 0
						freq_dict['fle'][suff] += 1

			if 'fem' in lb :
				for tk_span in tks_span :
					tk = tk_span[0].lower()
					if tk not in freq_dict['fem'] :
						freq_dict['fem'][tk] = 0
					freq_dict['fem'][tk] += 1
			if 'neu' in lb :
				for tk_span in tks_span :
					tk = tk_span[0].lower()
					if tk not in freq_dict['neu'] :
						freq_dict['neu'][tk] = 0
					freq_dict['neu'][tk] += 1
			if 'epi' in lb :
				for tk_span in tks_span :
					tk = tk_span[0].lower()
					if tk not in freq_dict['epi'] :
						freq_dict['epi'][tk] = 0
					freq_dict['epi'][tk] += 1


			for lb_indiv in lb :
				if ext not in ext_dict[lb_indiv] :
					ext_dict[lb_indiv][ext] = 0
				ext_dict[lb_indiv][ext] += 1

	for text in freq_dict['coo'] :
		freq_df['coo']['text'].append(text)
		freq_df['coo']['occ'].append(freq_dict['coo'][text]) 

	for text in freq_dict['fle'] :
		freq_df['fle']['end'].append(text)
		freq_df['fle']['signs'].append( ' '.join( sorted(list( set([ item for item in text if re.match(punct, item) ])))) )
		freq_df['fle']['occ'].append(freq_dict['fle'][text]) 

	for text in freq_dict['fem'] :
		freq_df['fem']['word'].append(text)
		freq_df['fem']['occ'].append(freq_dict['fem'][text]) 

	for text in freq_dict['epi'] :
		freq_df['epi']['word'].append(text)
		freq_df['epi']['occ'].append(freq_dict['epi'][text]) 

	for text in freq_dict['neu'] :
		freq_df['neu']['word'].append(text)
		freq_df['neu']['occ'].append(freq_dict['neu'][text]) 


	lb_ext_occ = []

	for key in freq_dict :
		if len(freq_dict[key]) > 0 :
			pd.DataFrame(freq_df[key]).to_csv(sur_script_dir + '/dataset/stats/' + key + '_stat.csv' , mode='w', index=False)
			for ext in ext_dict[key] :
				lb_ext_occ.append( (key, ext, ext_dict[key][ext])  )

	ext_df = pd.DataFrame( {'label' : [item[0] for item in lb_ext_occ] , 'ext' : [item[1] for item in lb_ext_occ] , 'occ' : [item[2] for item in lb_ext_occ]}  )
	ext_df.to_csv(sur_script_dir + '/dataset/stats/ext_stats.csv' , mode='w', index=False)




compute_stats()