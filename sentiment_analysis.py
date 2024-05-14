import json 
import tensorflow as tf

with open("sarcasm.json", 'r') as f: 
    datastore = json.load(f) #reading file and storing json 

sentences = []
labels = []
urls = []
for item in datastore: #iterating through all json items and storing items in respective lists
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
tokenizer_old = Tokenizer(oov_token = "<OOV>")
tokenizer_old.fit_on_texts(sentences)
word_index = tokenizer_old.word_index

sequences = tokenizer_old.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding="post")
# print(padded[0])
# print(padded.shape)
#prints: 
# [  308 15115   679  3337  2298    48   382  2576 15116     6  2577  8434
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0]
# (26709, 40)

#tokenizer should only see training data: 
training_size = 20000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

vocab_size = 10000
oov_tok = "<OOV>"
max_length = 100
trunc_type = 'post'
padding_type = 'post'

tokenizer = Tokenizer(num_words=vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = max_length, padding = "post", truncating = "post")

# print(testing_padded)

import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)


#embedding: 
embedding_dim = 16 #begin with a small value like 50-100 to allow model to learn, and then increase until you start seeing diminishing returns
#size roughly 10-20% of vocab size

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#training model: 
num_epochs = 30 #training length

history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

model.summary()

sentence = [
    "granny starting to fear spiders in the garden might be real",
    "the weather today is bright and sunny"
]

sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)

print(model.predict(padded))