import pickle
import sklearn
import numpy as np
from tensorflow import keras
from underthesea import pos_tag
from underthesea import word_tokenize


def encode_text(text):
    text = text.lower()
    text_format = text.translate(translator)
    text_split = word_tokenize(text_format)
    text_values = [dictionary.get(i, dictionary["<unknown>"]) for i in text_split]
    text_tags = [tags_dictionary.get(i[1].lower(), tags_dictionary["<unknown>"]) for i in pos_tag(text_format)]
    text_values = resize(text_values, 50, dictionary["<unknown>"])
    text_tags = resize(text_tags, 50, tags_dictionary["<unknown>"])
    return text_values, text_tags


def resize(array, new_size, default_value):
    array_size = len(array)
    if new_size > array_size:
        while array_size < new_size:
            array.append(default_value)
            array_size += 1
    else:
        array = array[:new_size]
    return array


with open("dictionary.pickle", "rb") as file:
    dictionary = pickle.load(file)

tags_dictionary = {
    "<unknown>": 0,
    "n": 11,
    "v": 12,
    "a": 13,
    "p": 14,
    "c": 15,
    "e": 16,
    "i": 17,
    "x": 18
}

translator = str.maketrans('', '', r"""!"#$%&'()*+,-./:;<=>?@”“[\]^_`{|}~""")

# load data
with open("train_data_sentences_1_value.pickle", "rb") as file:
    train_data_sentences_1_values = pickle.load(file)
with open("train_data_sentences_2_value.pickle", "rb") as file:
    train_data_sentences_2_values = pickle.load(file)
with open("train_data_sentences_1_tags.pickle", "rb") as file:
    train_data_sentences_1_tags = pickle.load(file)
with open("train_data_sentences_2_tags.pickle", "rb") as file:
    train_data_sentences_2_tags = pickle.load(file)
with open("train_data_labels.pickle", "rb") as file:
    train_data_labels = pickle.load(file)

# split data
train_s1, test_s1, train_s2, test_s2, train_t1, test_t1, train_t2, test_t2, train_lb, test_lb = sklearn.model_selection.train_test_split(
    train_data_sentences_1_values, train_data_sentences_2_values, train_data_sentences_1_tags,
    train_data_sentences_2_tags, train_data_labels, test_size=0.1)

# statement 1 CNN RNN output
statement_1_input = keras.layers.Input(shape=50)
statement_1_embedding = keras.layers.Embedding(40000, 16)(statement_1_input)

statement_1_convolution_1 = keras.layers.Conv1D(16, 5, padding='same', activation='relu')(statement_1_embedding)
statement_1_maxPool_1 = keras.layers.MaxPooling1D(2, padding="same")(statement_1_convolution_1)

statement_1_convolution_2 = keras.layers.Conv1D(16, 4, padding='same', activation='relu')(statement_1_embedding)
statement_1_maxPool_2 = keras.layers.MaxPooling1D(2, padding="same")(statement_1_convolution_2)

statement_1_concat = keras.layers.concatenate([statement_1_maxPool_1, statement_1_maxPool_2])
statement_1_output = keras.layers.LSTM(32, return_sequences=True, activation="sigmoid")(statement_1_concat)

# statement 2 CNN RNN output
statement_2_input = keras.layers.Input(shape=50)
statement_2_embedding = keras.layers.Embedding(40000, 16)(statement_2_input)

statement_2_convolution_1 = keras.layers.Conv1D(16, 5, padding='same', activation='relu')(statement_2_embedding)
statement_2_maxPool_1 = keras.layers.MaxPooling1D(2, padding="same")(statement_2_convolution_1)

statement_2_convolution_2 = keras.layers.Conv1D(16, 4, padding='same', activation='relu')(statement_2_embedding)
statement_2_maxPool_2 = keras.layers.MaxPooling1D(2, padding="same")(statement_2_convolution_2)

statement_2_concat = keras.layers.concatenate([statement_2_maxPool_1, statement_2_maxPool_2])
statement_2_output = keras.layers.LSTM(32, return_sequences=True, activation="sigmoid")(statement_2_concat)

# statement similarity output
similarity_merge = keras.layers.dot(([statement_1_embedding, statement_2_embedding]), axes=1)

similarity_convolution_1 = keras.layers.Conv1D(16, 5, padding="same", activation="relu")(similarity_merge)
similarity_maxPool_1 = keras.layers.MaxPool1D(2, padding="same")(similarity_convolution_1)

similarity_convolution_2 = keras.layers.Conv1D(16, 4, padding="same", activation="relu")(similarity_merge)
similarity_maxPool_2 = keras.layers.MaxPool1D(2, padding="same")(similarity_convolution_2)

similarity_output = keras.layers.concatenate(([similarity_maxPool_1, similarity_maxPool_2]))

# statement tags output
tags_1_input = keras.layers.Input(shape=50)
tags_1_embedding = keras.layers.Embedding(30, 5)(tags_1_input)

tags_convolution_1 = keras.layers.Conv1D(16, 5, padding="same", activation="relu")(tags_1_embedding)
tags_maxPool_1 = keras.layers.MaxPool1D(2, padding="same")(tags_convolution_1)

tags_2_input = keras.layers.Input(shape=50)
tags_2_embedding = keras.layers.Embedding(30, 5)(tags_2_input)

tags_convolution_2 = keras.layers.Conv1D(16, 4, padding="same", activation="relu")(tags_2_embedding)
tags_maxPool_2 = keras.layers.MaxPool1D(2, padding="same")(tags_convolution_2)

tags_output = keras.layers.concatenate(([tags_maxPool_1, tags_maxPool_2]))

# dense output
statement_subtract = keras.layers.subtract(([statement_1_output, statement_2_output]))
flatten_statement = keras.layers.Flatten()(statement_subtract)
flatten_similarity = keras.layers.Flatten()(similarity_output)
flatten_tags = keras.layers.Flatten()(tags_output)
dense_concat = keras.layers.concatenate(([flatten_statement, flatten_similarity, flatten_tags]))  #
dense_output_1 = keras.layers.Dense(16, activation="relu")(dense_concat)
dense_output_2 = keras.layers.Dense(16, activation="relu")(dense_output_1)
prediction = keras.layers.Dense(1, activation="sigmoid")(dense_output_2)

# model
model = keras.Model(inputs=([statement_1_input, statement_2_input, tags_1_input, tags_2_input]), outputs=[prediction])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# training
model.fit([train_s1, train_s2, train_t1, train_t2], [train_lb], batch_size=128, epochs=10)

# save
model.save("paraphrase_model_40.md5")
# model = keras.models.load_model("paraphrase_model_40.md5")

# evaluate
result = model.evaluate([test_s1, test_s2, test_t1, test_t2], [test_lb])
print(result)
# prediction = model.predict([test_s1, test_s2, test_t1, test_t2])
# print(prediction)
# print("done")

while True:
    sentence_1 = input("nhập câu 1: ")
    sentence_2 = input("nhập câu 2: ")

    index_1, tag_1 = encode_text(sentence_1)
    index_2, tag_2 = encode_text(sentence_2)

    test_index_1 = [[]]
    test_index_2 = [[]]
    test_tag_1 = [[]]
    test_tag_2 = [[]]

    test_index_1[0] = np.asarray(index_1)
    test_index_2[0] = np.asarray(index_2)
    test_tag_1[0] = np.asarray(tag_1)
    test_tag_2[0] = np.asarray(tag_2)

    predict = model.predict([test_index_1, test_index_2, test_tag_1, test_tag_2])
    print(predict)
    print("done")
