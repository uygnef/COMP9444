import tensorflow as tf
import numpy as np
import glob  # this will be useful when reading reviews from file
import os
import tarfile

batch_size = 40


def load_data(glove_dict):
    """
    Take reviews from text files, vectorized them, and load them into a
    numpy array. Any prepossessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    pos_files = glob.glob("../pos/*.txt")
    neg_files = glob.glob("../neg/*.txt")
    vect_list = []
    import string
    for file_name in pos_files + neg_files:
        data = open(file_name, 'r', encoding='utf-8').readline().split()
        # TODO: stop words, word cleaning...
        i = 0
        index_array = [None for _ in range(40)]
        while i < 40:
            try:
                index_array[i] = glove_dict[data[i].lower().replace(string.punctuation, "")]
            except IndexError:
                index_array[i] = 0
            except KeyError:
                index_array[i] = 0
            except AttributeError:
                index_array[i] = 0
            i += 1
        vect_list.append(index_array)
    return vect_list

def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    # data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    # if you are running on the CSE machines, you can load the glove data from here
    # data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    raw_data = open("glove.6B.50d.txt", 'r', encoding="utf-8")
    vec_array = [[0 for _ in range(40)]]
    word_index_dict = {"UNK" : 0}

    for i, line in enumerate(raw_data):
        word, array = line.split(" ", 1)
        word_index_dict[word] = i + 1
        print(len(line.split()[1:]))
        vec_array.append(map(float, line.split()[1:]))

    embeddings = np.array(vec_array)
    return embeddings, word_index_dict
e, c = load_glove_embeddings()
print(e)
a = load_data(c)


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""

    return input_data, labels, optimizer, accuracy, loss





