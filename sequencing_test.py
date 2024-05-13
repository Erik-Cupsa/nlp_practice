import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

sentences = [ #input sentences
    'i love my dog',
    'I, love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

# tokenizer = Tokenizer(num_words = 100) ## will only keep the 100 most frequent words
tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>") #token that you do not expect to see in corpus
tokenizer.fit_on_texts(sentences) #updates internal vocabulary with sentences
word_index = tokenizer.word_index #dictionary where keys are words, and values are index
sequences = tokenizer.texts_to_sequences(sentences) #converts sentences to sequences of characters, with each word replaced by its index
padded = pad_sequences(sequences) #adds 0s to front in shorter sentences
#can do pad_sequences(sequences, padding='post', truncating = 'post', maxlen = 5) to add 0s at end, or specify max length
print(word_index) #{'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
print(sequences) #[[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
print(padded) #prints below: 
# [[ 0  0  0  5  3  2  4]
#  [ 0  0  0  5  3  2  7]
#  [ 0  0  0  6  3  2  4]
#  [ 8  6  9  2  4 10 11]]
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq) #[[4, 2, 1, 3], [1, 3, 1]]
