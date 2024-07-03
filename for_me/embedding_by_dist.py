import numpy as np
import matplotlib.pyplot as plt
import math
import random

def euclidian_distance(a, b) :
	return math.sqrt( ((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2)  )


def loss_function(points, dists) :
	somme = 0
	for i in range(0, len(points)) :
		for j in range(i+1, len(points)) :
			somme += ( ((points[i][0] - points[j][0]) ** 2) + ( (points[i][1] - points[j][1]) ** 2 ) - dists[i][j] ) ** 2
	return somme

def loss_gradiant(points, dists) :
	grad = []
	for i in range(0, len(points)) : 
		dx = sum( (points[i][0] - points[j][0]) * ( ((points[i][0] - points[j][0]) ** 2) + ( (points[i][1] - points[j][1]) ** 2 ) - dists[i][j] ) for j in range(0,len(points)) )
		dy = sum( (points[i][1] - points[j][1]) * ( ((points[i][0] - points[j][0]) ** 2) + ( (points[i][1] - points[j][1]) ** 2 ) - dists[i][j] ) for j in range(0,len(points)) )
		grad.append( (dx,dy) )
	return grad

def gradiant_descent(points, dists, epsilon=1) :
	grad = loss_gradiant(points, dists)
	return [  ( points[i][0] - epsilon * grad[i][0], points[i][1] - epsilon * grad[i][1]  ) for i in range(0, len(points))  ]


def greedy_aligned(data, dists, xlims=(-100, 100), ylims=(-100,100) ) :

	res = [None] * len(data)

	maxval = None
	argmax = None
	for i in range(0, len(data)) :
		for j in range(i+1, len(data)) :
			if maxval is None or maxval < dists[i][j] :
				maxval = dists[i][j]
				argmax = (i,j)

	done = [argmax[0], argmax[1]]
	res[argmax[0]] = (xlims[0], ylims[0] )
	res[argmax[1]] = (xlims[1], ylims[1] )
	left = [x for x in range(0,len(data)) if x not in done]
	while len(left) > 0 :
		minval = None
		argmin = None
		for i in range(0, len(done)) :
			for j in range(i+1, len(done)) :
				for k in left :
					i_val = done[i]
					j_val = done[j]
					if minval is None or abs( dists[k][i_val] - dists[k][j_val] ) < minval :
						argmin = (i_val,j_val,k)

		delta = dists[done[0]][k] / (dists[done[0]][k]  + dists[done[1]][k] )
		res[k] = (  (res[i_val][0] + res[j_val][0]) / 2 , ylims[0] + ( delta  * (ylims[1] - ylims[0]) ) )
		left.remove(k)
		done.append(k)

	return res


def greedy_embedding(data, dists, xlims=(-100, 100), ylims=(-100,100), goal=10, epsilon=0.000005, max_iter=1000, output_filename='embed.png') : 

	left = list(range(0,len(data)))
	done = []

	res = greedy_aligned(data, dists, xlims=xlims, ylims=ylims)
	loss = goal + 1
	cpt = 0

	while cpt < max_iter and loss > goal :
		cpt += 1
		res = gradiant_descent(res, dists, epsilon=epsilon)
		loss = loss_function(res, dists)
		print(f'loss={loss}')
	
	fig, ax = plt.subplots()
	ax.scatter( [e[0] for e in res], [e[1] for e in res])
	for i in range(0,len(res)) :
		ax.annotate( str(data[i]) , res[i] )

	nxlims = ( min([item[0] for item in res]) , max([item[0] for item in res])  )
	nylims = ( min([item[1] for item in res]) , max([item[1] for item in res])  )

	ax.set_xlim(nxlims[0], nxlims[1])
	ax.set_ylim(nylims[0], nylims[1])
	plt.savefig(output_filename)
	return res 





