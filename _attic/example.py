import sys
import os
script_dir = os.path.dirname( os.path.abspath(__file__) )
sys.path.append(os.path.dirname(script_dir))

from etude_fr_inclusif.super_model import SuperAnnModel
from etude_fr_inclusif.adv_rule_based import AdvRBModel
from etude_fr_inclusif.inclure import InclureModel

"""
An example of prediction by a model
"""

model = SuperAnnModel([AdvRBModel(), InclureModel()], tol=0.8)

text = "Je vous souhaite à toutes les infirmières et tous les infirmiers, les étudiants sont là, l'écrivaine aussi."

print(model.predict(text))
