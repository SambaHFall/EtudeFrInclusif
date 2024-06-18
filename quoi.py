from etude_fr_inclusif.crf import CRFModel
import spacy
import codecs
from colorama import Fore, Style

model = CRFModel()

text = "Ceci (est) un petit test de ce qui pourrait arriv(er) (on teste plusieurs trucs)"

doc = model.nlp_model(text)

for tk in doc:
	print(tk.text)
	print(tk.lemma_)
	print(tk.pos_)
	print("---")
