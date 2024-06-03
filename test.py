from src._utils import *
from sklearn.model_selection import train_test_split
from src.naive_rule_based import NaiveRBModel
from src.crf import CRFModel
from src.inclure import InclureModel
from src.adv_rule_based import AdvRBModel
from src.adv_rule_based import *
from src.super_model import SuperAnnModel

import matplotlib.pyplot as plt
import numpy as np


x, y = get_fr_inclusif_data()

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.2, random_state=42)

models_list = []
weights_list = []


modelARB = AdvRBModel(proc=['fle', 'coo', 'neu', 'fem', 'epi'])
models_list.append(modelARB)

preds = modelARB.predict(x)

print("Métriques pour rule-based adv :")
print(modelARB.metric(x, y, preds))
weights_list.append(modelARB.metric(x, y, preds)["precision"])

modelARB.annotation_layout(x, preds, iddoc=range(0,len(x))).to_csv("res/arb_preds.csv")
modelARB.annotation_layout(x,y,iddoc=range(0,len(x))).to_csv("res/obs.csv")


modelNRB = NaiveRBModel()
models_list.append(modelNRB)

preds = modelNRB.predict(x)

print("Métriques pour rule-based naïf :")
print(modelNRB.metric(x, y, preds))
weights_list.append(modelNRB.metric(x, y, preds)["precision"])

modelNRB.annotation_layout(x, preds, iddoc=range(0,len(x))).to_csv("res/nrb_preds.csv")

modelCRF = CRFModel()

modelCRF.fit(train_x, train_y)

preds = modelCRF.predict(test_x)

print("Métriques pour CRF :")
print(modelCRF.metric(test_x, test_y, preds))

modelCRF.annotation_layout(test_x, preds, iddoc=range(0,len(test_x))).to_csv("res/crf_preds.csv")

modelINC = InclureModel()
models_list.append(modelINC)

preds = modelINC.predict(x)

print("Métriques pour Inclure :")
print(modelINC.metric(x, y, preds))
weights_list.append(modelINC.metric(x, y, preds)["precision"])

modelINC.annotation_layout(x, preds, iddoc=range(0,len(x))).to_csv("res/inclure_preds.csv")


gran = 0.01
x_axis = [ item * gran for item in np.arange(0, 1, gran) ]
y_axes = {"precision" : [], "recall" : [], "f1-score" : []}
for j in range(0,len(x_axis)) :
	x_val = x_axis[j]
	modelSUPER = SuperAnnModel(models = models_list, weights = weights_list, tol= x_val)
	preds = modelSUPER.predict(x)
	for key in y_axes :
		y_axes[key].append(modelSUPER.metric(x,y,preds)[key])

fig, ax = plt.subplots()

for key in y_axes :
	ax.plot(x_axis, y_axes[key])

ax.legend(labels=y_axes.keys())
plt.savefig('super_model_perf.png')
