import tensorflow as tf
import pickle
import sklearn
import pickle
from underthesea import word_tokenize
from underthesea import pos_tag
import numpy as np
from tensorflow import keras


class MyTextEncoder:

    def __init__(self):
        with open("dictionary.pickle", "rb") as file:
            self.dictionary = pickle.load(file)

        self.tags_dictionary = {
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

        self.translator = str.maketrans('', '', r"""!"#$%&'()*+,-./:;<=>?@”“[\]^_`{|}~""")

    def encode(self, text):

        text = text.lower()
        text_format = text.translate(self.translator)
        text_split = word_tokenize(text_format)
        text_values = [self.dictionary.get(i, self.dictionary["<unknown>"]) for i in text_split]
        text_tags = [self.tags_dictionary.get(i[1].lower(), self.tags_dictionary["<unknown>"]) for i in
                     pos_tag(text_format)]
        text_values = self.resize(text_values, 50, self.dictionary["<unknown>"])
        text_tags = self.resize(text_tags, 50, self.tags_dictionary["<unknown>"])
        return text_values, text_tags

    def resize(self, array, new_size, default_value):
        array_size = len(array)
        if new_size > array_size:
            while array_size < new_size:
                array.append(default_value)
                array_size += 1
        else:
            array = array[:new_size]
        return array
