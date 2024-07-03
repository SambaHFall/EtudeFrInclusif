import os

script_dir = os.path.dirname( os.path.abspath(__file__) )
sur_script_dir = os.path.dirname(script_dir)

import pandas as pd

"""
Produce a same_entry.txt file, which enumerate all the group of entry in the dataset that are similar
"""

class CorpusEntryCluster() :

	def __init__(self, egal=(lambda x,y : x == y) ) :
		self.texts_clusters = []
		self.sub_clusters = []
		self.egal = egal

	def add(self, entry) :
		found = False
		sid = entry["sample_id"]
		text = entry["text"]
		anns = str(entry["spans"])
		for ind in range(0, len(self.texts_clusters) ) :
			if self.egal(text, self.texts_clusters[ind][0]) :
				self.texts_clusters[ind].append(sid)
				sub_found = False
				for sub_clust in self.sub_clusters[ind] :
					if sub_clust[0] == anns :
						sub_clust.append(sid)
						sub_found = True
				if not sub_found :
					self.sub_clusters[ind].append([anns, sid])
				found = True
				break
		if not found :
			self.texts_clusters.append([text, sid])
			self.sub_clusters.append( [ [anns, sid] ] )

	def remove_too_small(self, neglig=1) :
		ind = 0
		while ind < len(self.texts_clusters) :
			if len(self.texts_clusters[ind]) <= (1 + neglig) :
				self.texts_clusters.pop(ind)
				self.sub_clusters.pop(ind)
			else :
				ind += 1

	def export(self, integral_path="./same_entry.txt") :

		with open(integral_path, 'w') as f :
			for ind in range(0, len(self.texts_clusters)) :
				if ind > 0 :
					f.write("\n---\n\n")
				texts_clust = self.texts_clusters[ind]
				f.write(f"Cluster '{texts_clust[0][:10]}...' ({len(texts_clust) - 1}):\n")
				for sub_clust in self.sub_clusters[ind] :
					line = ' | '.join( sub_clust[1:] )
					f.write(f'- {sub_clust[0]} : {line}\n')




def count_same_entry(input_filename="new_corpus.csv") :

	input_df = pd.read_csv(sur_script_dir + "/dataset/" + input_filename)

	similar_clusters = CorpusEntryCluster()

	for i, content in input_df.iterrows() :
		similar_clusters.add(content)

	similar_clusters.remove_too_small()

	similar_clusters.export(integral_path = sur_script_dir + "/dataset/same_entry.txt")


count_same_entry()