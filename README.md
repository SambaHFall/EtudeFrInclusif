# EtudeFrInclusif

This project provides methods to detect inclusive french within document in french.
It contains for now 4 different inclusive-french-annotation models : 3 of them are rule-based, while the last one is a statistical model working with CRF. 

## Models

### AnnPredModel
It is an interface for annotation models used for all the following models. It deals with Ann, a class implemented in this project to represent annotation in a document. It has 4 attributes : beg (the index of the character where the annotation starts in the document), end (the index of the first character after the annotation in the document), text (the part of the document that is annotated) and metadata (a dict that can contain any kind of metadata that the user feel relevant to store).
Every AnnPredModel has two main functions : fit() used to train the model (if necessary), and predict used to make predictions. Both deals with a list of str, and a list of list of Ann to represent documents and their respective annotations.

### NaiveRBModel 
A rule-based method model using naive rules dealing with basic recognizing patterns or special characters in words or sentences that are usual to inclusive french words.

### AdvRBModel
A rule-based method model built after the NaiveRBModel model, with rules aimed to be more precise than NaiveRBModel's.

### CRFModel
A statistical model working with CRF and a personnalized representation of words.

### InclureModel
A rule-based method model using the same identification method as in the project INCLURE ( https://github.com/PaulLerner/inclure ).

## Super Model
The SuperAnnModel model is a voting system model which bases its predictions on the predictions of severals simple models : for every token in a text, it computes an average prediction score based on what the simple models predicted (the prediction of each model being balanced by a certain weight, representing the trust put in that model), then it considers the prediction as a YES (or a NO) whether the score is greater than a certain threshold.


# TODO

installer fr_core_news_sm spacy

# Exemple de code
```python
import ....

model = supermodel()

print(model.predict( ["Ceci est un(e) test de qualité(s) supérieur(e)s pour les auteurs et autrices de ce texte"] ))

```
