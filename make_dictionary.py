import pandas as pd
import numpy as np
import pickle

dictionary_csv_file = pd.read_csv("VDic_uni.csv", sep=';', usecols=['Từ']).to_numpy()

dictionary = {
    "<unknown>": 0,
}

index = 10
for word in dictionary_csv_file:
    dictionary[str(word[0]).lower()] = index
    index += 1

with open("dictionary.pickle", "wb") as file:
    pickle.dump(dictionary, file)

print("done")
