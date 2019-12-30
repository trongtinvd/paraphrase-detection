import pickle
from underthesea import word_tokenize
from underthesea import pos_tag
import numpy as np
from MyTextEncoder import  MyTextEncoder

'''
def encode_text(text):
    text = text.lower()
    text_format = text.translate(translator)
    text_split = word_tokenize(text_format)
    text_values = [dictionary.get(i, dictionary["<unknown>"]) for i in text_split]
    text_tags = [tags_dictionary.get(i[1].lower(), tags_dictionary["<unknown>"])for i in pos_tag(text_format)]
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
'''
myTextEncoder = MyTextEncoder()

sentences_1_values = []
sentences_2_values = []
sentences_1_tags = []
sentences_2_tags = []
labels = []

with open("vnPara/Sentences1.txt", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()
    for line in lines:
        values, tags = myTextEncoder.encode_text(line)
        sentences_1_values.append(np.asarray(values))
        sentences_1_tags.append(np.asarray(tags))

with open("vnPara/Sentences2.txt", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()
    for line in lines:
        values, tags = myTextEncoder.encode_text(line)
        sentences_2_values.append(np.asarray(values))
        sentences_2_tags.append(np.asarray(tags))

with open("vnPara/Labels.txt", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()
    for line in lines:
        labels.append(int(line))

sentences_1_values = np.asarray(sentences_1_values)
sentences_2_values = np.asarray(sentences_2_values)
sentences_1_tags = np.asarray(sentences_1_tags)
sentences_2_tags = np.asarray(sentences_2_tags)
labels = np.asarray(labels)

with open("train_data_sentences_1_value.pickle", "wb") as file:
    pickle.dump(sentences_1_values, file)

with open("train_data_sentences_2_value.pickle", "wb") as file:
    pickle.dump(sentences_2_values, file)

with open("train_data_sentences_1_tags.pickle", "wb") as file:
    pickle.dump(sentences_1_tags, file)

with open("train_data_sentences_2_tags.pickle", "wb") as file:
    pickle.dump(sentences_2_tags, file)

with open("train_data_labels.pickle", "wb") as file:
    pickle.dump(labels, file)


print("done")
