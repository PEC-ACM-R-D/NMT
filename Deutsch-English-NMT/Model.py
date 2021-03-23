from Attention import AttentionDecoder
from Utils import load_clean_sentences
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
tokenizer = Tokenizer()


def create_tokenizer(lines):
    tokenizer.fit_on_texts(lines)
    return tokenizer


def max_length(lines):
    return max(len(line.split()) for line in lines)

# encode and pad sequences


def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


def encode_output(sequences, vocab_size):
    ylist = []
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


dataset = load_clean_sentences('english_german-both.pkl')
train = load_clean_sentences('english_german-train.pkl')
test = load_clean_sentences('english_german-test.pkl')

# ENG TOKENIZER
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])

# GER TOKENIZER
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])

trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)

testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)

encoder_inputs = Input(shape=(None, ger_length))
encoder = LSTM(256, return_state=True)
encoder_outputs, states_h, states_c = encoder(encoder_inputs)
encoder_states = (states_h, states_c)

decoder_inputs = Input(shape=(None, eng_length))
decoder_LSTM = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_LSTM(
    decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(eng_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([trainX, testX], testX, batch_size=32, epochs=50)
