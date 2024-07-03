from etude_fr_inclusif.adv_rule_based import AdvRBModel
from etude_fr_inclusif.inclure import InclureModel
from etude_fr_inclusif.super_model import SuperAnnModel

"""
Test code for the etude_fr_inclusif package
"""

model = SuperAnnModel([AdvRBModel(), InclureModel()], tol=0.8)

text = "Je vous souhaite à toutes les infirmières et tous les infirmiers, les étudiants sont là, l'écrivaine aussi."

print(model.predict(text))

