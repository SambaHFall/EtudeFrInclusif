import sys
import os
script_dir = os.path.dirname( os.path.abspath(__file__) )
sur_script_dir = os.path.dirname(script_dir)
sys.path.append(sur_script_dir)

from etude_fr_inclusif._utils import AnnPredModel, get_fr_inclusif_data
from etude_fr_inclusif.splitter import train_test_splitter
from etude_fr_inclusif.naive_rule_based import NaiveRBModel
from etude_fr_inclusif.crf import CRFModel
from etude_fr_inclusif.inclure import InclureModel
from etude_fr_inclusif.adv_rule_based import AdvRBModel
from etude_fr_inclusif.super_model import SuperAnnModel

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

x, y = get_fr_inclusif_data()

train_x, test_x, train_y, test_y = train_test_splitter(x,y, test_size=0.2)

models_list = []
weights_list = []


modelARB = AdvRBModel(proc=['fle', 'coo', 'neu', 'fem'])
models_list.append(modelARB)

preds = modelARB.predict(test_x)

print("Métriques pour rule-based adv :")
print(modelARB.metric(test_x, test_y, preds))
weights_list.append(modelARB.metric(test_x, test_y, preds)["precision"])

modelARB.annotation_layout(test_x, preds, iddoc=range(0,len(test_x))).to_csv(sur_script_dir + "/res/arb_preds.csv")
modelARB.annotation_layout(test_x, test_y ,iddoc=range(0,len(test_x))).to_csv(sur_script_dir + "/res/obs.csv")


modelNRB = NaiveRBModel()
models_list.append(modelNRB)

preds = modelNRB.predict(test_x)

print("Métriques pour rule-based naïf :")
print(modelNRB.metric(test_x, test_y, preds))
weights_list.append(modelNRB.metric(test_x, test_y, preds)["precision"])

modelNRB.annotation_layout(test_x, preds, iddoc=range(0,len(test_x))).to_csv(sur_script_dir + "/res/nrb_preds.csv")

modelCRF = CRFModel()

modelCRF.fit(train_x, train_y)

preds = modelCRF.predict(test_x)

print("Métriques pour CRF :")
print(modelCRF.metric(test_x, test_y, preds))

modelCRF.annotation_layout(test_x, preds, iddoc=range(0,len(test_x))).to_csv(sur_script_dir + "/res/crf_preds.csv")

modelINC = InclureModel()
models_list.append(modelINC)

preds = modelINC.predict(test_x)

print("Métriques pour Inclure :")
print(modelINC.metric(test_x, test_y, preds))
weights_list.append(modelINC.metric(test_x, test_y, preds)["precision"])

modelINC.annotation_layout(test_x, preds, iddoc=range(0,len(test_x))).to_csv(sur_script_dir + "/res/inclure_preds.csv")


gran = 0.05
x_axis = [ item for item in np.arange(0, 1+gran, gran) ]
y_axes = {"precision" : [], "recall" : [], "f1-score" : []}
for j in range(0,len(x_axis)) :
	x_val = x_axis[j]
	modelSUPER = SuperAnnModel(models = models_list, weights = weights_list, tol= x_val)
	preds = modelSUPER.predict(test_x)
	for key in y_axes :
		y_axes[key].append(modelSUPER.metric(test_x,test_y,preds)[key])

fig, ax = plt.subplots()

for key in y_axes :
	ax.plot(x_axis, y_axes[key])


ax.set_xlabel("tolerance")
ax.set_ylabel("score")

best_ind = np.argmax([.0 if item is None else item for item in y_axes["f1-score"]])
best_x = x_axis[best_ind]

ax.text(best_x, 0.95, 
	'Meilleur f1-score : \n tol=' + str(round(best_x,2)) + '\n precision=' + str(round(y_axes['precision'][best_ind], 4)) + '\n recall=' + str( round(y_axes['recall'][best_ind],4) ) + '\n f1-score=' + str(round(y_axes['f1-score'][best_ind], 4) ),
	bbox = {'facecolor' : 'white', 'alpha' : 0.4, 'pad' : 10} )

ax.legend(labels=y_axes.keys())

plt.savefig(sur_script_dir + '/res/super_model_perf.png')
