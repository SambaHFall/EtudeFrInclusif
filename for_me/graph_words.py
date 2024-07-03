import networkx as nx
import matplotlib.pyplot as plt

def facts(wd) :
	if len(wd) <= 1 :
		return [wd]
	else :
		res = []
		for i in range(1, len(wd)) :
			res.append(wd[0:i])
		return res + facts(wd[1:])


def is_included(wd1, wd2) :
	return wd1 in facts(wd2)


def longest_common_fact(wd1, wd2) :
	common = list(set( facts(wd1) ).intersection( set( facts(wd2) ) ))
	return max( [len(item) for item in common] ) if len(common) > 0 else None



def graph_visualisation(wds, sizes, output_filename='graph.png', min_fact=2) :
	graph = nx.DiGraph()
	graph.add_nodes_from(wds)

	for i in range(0, len(wds)):
		for j in range(i+1, len(wds)) :
			wd1 = wds[i]
			wd2 = wds[j]
			if is_included(wd1, wd2) :
				graph.add_weighted_edges_from( [(wd1, wd2, len(wd1) / len(wd2) )], color='red' )
			else :
				d = longest_common_fact(wd1, wd2)
				if d is not None and d >= min_fact :
					graph.add_weighted_edges_from( [(wd1, wd2, d / len(wd1) ), (wd2, wd1, d / len(wd2) )], color='blue' )


	fig, ax = plt.subplots()
	nx.draw_networkx(graph, with_labels=True, node_size=sizes, edge_color= [ graph[u][v]['color'] for u,v in graph.edges() ], width=[ graph[u][v]['weight'] for u,v in graph.edges() ] )
	plt.savefig(output_filename)



