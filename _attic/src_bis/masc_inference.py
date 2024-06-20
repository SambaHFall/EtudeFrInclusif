from simpletransformers.seq2seq import Seq2SeqModel
import os
import numpy as np
import random
from collections import Counter
from keras.models import Model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from tqdm import tqdm

from sklearn.neural_network import MLPClassifier


def uniformizer(arr, n, filler=0):
	res = arr.copy()
	while len(res) < n:
		res.append(filler)
	return res

def convert_word_to_vect(wd, n) :
	return uniformizer([ord(c) for c in wd], n)


def convert_vect_to_word(vect, filler=0) :
	res = ""
	cpt = 0
	while cpt < len(vect) and vect[cpt] != filler :
		res += chr(vect[cpt])
		cpt += 1
	
	return res
 

def create_model(input_size = 21, output_size = 21, vocab=1000, lstm_size = 64, embedding_dim=300) :
	input_sequence = Input(shape=(input_size,))
	embedding = Embedding(input_dim=vocab, output_dim=embedding_dim,)(input_sequence)
	encoder = LSTM(lstm_size, return_sequences=False)(embedding)
	r_vec = RepeatVector(output_size)(encoder)
	decoder = LSTM(lstm_size, return_sequences=True, dropout=0.2)(r_vec)
	logits = TimeDistributed(Dense(vocab))(decoder)

	enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
	enc_dec_model.compile(loss=sparse_categorical_crossentropy,
		optimizer=Adam(1e-3),
		metrics=['accuracy'])
	return enc_dec_model



def masc_inference_model() : 

	parent_path = os.path.dirname(os.path.dirname(__file__) )

	raw_data = {}

	input_max_length = 0
	output_max_length = 0

	with open(parent_path + '/data/flexrulesuni.txt') as f : 
		lines = f.readlines()
		for line in tqdm(lines) :
			parts = line.split('\t')
			raw_data[parts[1]] = parts[0] 
			input_max_length = max(input_max_length, len(parts[1]) )
			output_max_length = max(output_max_length, len(parts[0]) )

	print("Data loaded (1/2)")

	raw_data_fem = raw_data.copy()
	sz = 2 * len(raw_data)

	with open(parent_path + '/data/dictrules.txt') as f : 
		lines = f.readlines()
		random.shuffle(lines)
		cpt = 0
		for line in tqdm(lines) :
			wd = line[:-1]
			if wd not in raw_data and len(wd.split(' ')) == 1 :
				raw_data[wd] = wd
				input_max_length = max(input_max_length, len(wd) )
				output_max_length = max(output_max_length, len(wd) )
				cpt += 1
				if cpt >= sz :
					break

	print("Data loaded (2/2)")

	raw_items_fem = [(k,v) for k,v in raw_data_fem.items() ]
	raw_items_total = [(k, 0 if k == v else 1) for k,v in raw_data.items() ]

	inp = [convert_word_to_vect(it[0], input_max_length) for it in raw_items_fem]
	out = [convert_word_to_vect(it[1], output_max_length) for it in raw_items_fem]

	inp = np.array([np.array(item) for item in inp])
	inp = inp.reshape(*inp.shape, 1)
	out = np.array([np.array(item) for item in out])
	out = out.reshape(*out.shape, 1)


	inp_total = [convert_word_to_vect(it[0], input_max_length) for it in raw_items_total]
	out_total = [it[1] for it in raw_items_total]

	print("Data shaped")

	inf_model = create_model(input_size=input_max_length, output_size=output_max_length)

	inf_model.fit(inp, out, batch_size=30, epochs=200)

	print("Inference model trained")

	detect_model = MLPClassifier(random_state=42, max_iter=1000).fit(inp_total, out_total)

	print("Detection model trained")

	return inf_model, detect_model, input_max_length


def predict(inf_model, detect_model, n, wd) :

	test = detect_model.predict( [ convert_word_to_vect(wd, n) ] )[0]

	if test < 1 :
		return None


	x = np.array([np.array( convert_word_to_vect(wd,n) )])
	x = x.reshape(*x.shape, 1)

	pred = inf_model.predict(x)[0]
	return convert_vect_to_word( [ np.argmax(pred[i]) for i in range(pred.shape[0]) ] )


inf_model, detect_model, n = masc_inference_model()

ent = input("Entrer un mot féminin : ")

while ent != "STOP":
	print("prediction  : " + str(predict(inf_model, detect_model, n, ent) ) )
	ent = input("Entrer un mot féminin : ")