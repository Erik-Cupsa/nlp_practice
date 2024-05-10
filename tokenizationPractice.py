import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore

sentences = [ #input sentences
    'i love my dog',
    'I, love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words = 100) ## will only keep the 100 most frequent words
tokenizer.fit_on_texts(sentences) #updates internal vocabulary with sentences
word_index = tokenizer.word_index #dictionary where keys are words, and values are index
print(word_index) # {'love': 1, 'my': 2, 'i': 3, 'dog': 4, 'cat': 5, 'you': 6}
