import pandas as pd
import os

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""
produce a 'filter_exploration.csv' file from the original 'exploration.csv' file, by removing all the annotation which match with certain categories
"""

def filter_proc(input_filename = 'exploration.csv', proc_to_filter = ['fle']) :

	df = pd.read_csv( dir_path + '/dataset/' + input_filename )

	rows_to_remove = []

	for ind, row in df.iterrows() :
		categories = eval(row['category'])
		categories = set([item.split('_')[0] for item in categories])
		left_categories = list(categories.difference(proc_to_filter))

		if len(left_categories) == 0 :
			rows_to_remove.append(ind)

	output_filename = 'filter_'

	df.drop(rows_to_remove).to_csv(dir_path + '/dataset/filtered_' + input_filename, mode='w', index=False )


filter_proc()

