from src._utils import *
import random


def abs_dist(dictA, dictB):
	s = 0
	for key in dictA:
		val = dictA[key]
		if key in dictB :
			val = val - dictB[key]
		s += abs(val)
	for key in dictB:
		if key not in dictA:
			s += dictB[key]
	return s

def category_count(lann):
	res = {}
	for ann in lann:
		if "category" in ann.metadata :
			for cat in ann.metadata["category"]:
				if cat not in res :
					res[cat] = 0
				res[cat] += 1
	return res

def sum_category_counter(ctrA, ctrB) :
	res = {}
	for key in ctrA:
		res[key] = ctrA[key]
	for key in ctrB :
		if key not in res:
			res[key] = ctrB[key]
		else:
			res[key] += ctrB[key]
	return res

def train_test_splitter(xdata, ydata, test_size=0.2) :
	category_dict = {}
	for item in ydata:
		category_dict = sum_category_counter(category_dict, category_count(item))

	goal = {}
	for key in category_dict:
		goal[key] = category_dict[key] * test_size

	cur = {}

	inds = [item for item in range(0, len(xdata) ) ]
	random.shuffle(inds)

	xtrain = []
	xtest = []
	ytrain = []
	ytest= []

	for ind in inds:
		tmp = sum_category_counter(cur, category_count(ydata[ind]) )
		if abs_dist(tmp, goal) < abs_dist(cur, goal) :
			xtest.append(xdata[ind])
			ytest.append(ydata[ind])
			cur = tmp
		else :
			xtrain.append(xdata[ind])
			ytrain.append(ydata[ind])

	return xtrain, xtest, ytrain, ytest



x, y = get_fr_inclusif_data()

train_x, test_x, train_y, test_y = train_test_splitter(x,y)