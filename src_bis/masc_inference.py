from simpletransformers.seq2seq import Seq2SeqModel
import os
import numpy as np
from collections import Counter
from keras.models import Model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


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

	raw_inp = []
	raw_out = []

	input_max_length = 0
	output_max_length = 0

	with open(parent_path + '/data/flexrulesuni.txt') as f : 
		lines = f.readlines()
		for line in lines :
			parts = line.split('\t')
			raw_inp.append(parts[1])
			raw_out.append(parts[0])
			input_max_length = max(input_max_length, len(parts[1]) )
			output_max_length = max(output_max_length, len(parts[0]) )

	inp = [convert_word_to_vect(wd, input_max_length) for wd in raw_inp]
	out = [convert_word_to_vect(wd, output_max_length) for wd in raw_out]

	inp = np.array([np.array(item) for item in inp])
	inp = inp.reshape(*inp.shape, 1)
	out = np.array([np.array(item) for item in out])
	out = out.reshape(*out.shape, 1)


	model = create_model(input_size=input_max_length, output_size=output_max_length)

	model.fit(inp, out, batch_size=30, epochs=500)

	return model, input_max_length


def predict(model, n, wd) :
	x = np.array([np.array( convert_word_to_vect(wd,n) )])
	x = x.reshape(*x.shape, 1)

	pred = model.predict(x)[0]
	return convert_vect_to_word( [ np.argmax(pred[i]) for i in range(pred.shape[0]) ] )


model, n = masc_inference_model()

ent = input("Entrer un mot féminin : ")

while ent != "STOP":
	print("prediction  : " + predict(model, n, ent) )
	ent = input("Entrer un mot féminin : ")