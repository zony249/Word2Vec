
from curses.ascii import isalpha
from tkinter import N
import numpy as np
from datasets import load_dataset
import re
from tqdm import tqdm
from pprint import pprint
import random
import pickle
import itertools




class Word2Vec:
    def __init__(self, ds, n_dim=256):
        
        self.word_count = self.count_words(ds)
        self.word_count = self.filter_top(self.word_count, num=10000)
        self.n_dim = n_dim

        # {word: o, c}
        self.word_map = self.init_vec(self.word_count, n_dim=n_dim)
    
    def __getitem__(self):
        pass

    def count_words(self, ds, num_articles=1000):
        """ raw word count
        
        """
        iterds = iter(ds["train"])

        word_count = {}

        i=0

        for x in tqdm(iterds, "Parsing lines"):
            no_punc = re.sub("[^\w\s]", "", x["text"])
            words = no_punc.lower().split()

            for word in words:
                # print(word)
                if (word not in word_count):
                    if word.isalpha():
                        word_count[word] = 1
                else:
                    word_count[word] += 1
            if i >= num_articles:
                break
            i+=1
    
        return word_count

    def filter_top(self, word_count, num=10000):
        """ Gets the top X most frequent words        
        """

        sorted_dict = {}

        sorted_keys = sorted(word_count, key=word_count.get, reverse=True)
        for w in sorted_keys:
            sorted_dict[w] = word_count[w]

        first_bunch = {}
        for i, key in enumerate(sorted_keys):
            first_bunch[key] = sorted_dict[key]
            if i >= num:
                break

        return first_bunch

    def init_vec(self, word_count, n_dim=256):
        vecs = {}
        for key in tqdm(word_count, "Generating Vectors"):
            o = np.random.rand(1, n_dim)
            o /= np.sqrt(np.dot(o, o.T))

            c = np.random.rand(1, n_dim)
            c /= np.sqrt(np.dot(c, c.T))

            vecs[key] = [o, c]
        return vecs

    def save(self, path):

        with open(path, "wb") as f:
            pickle.dump(self.word_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as f:
            self.word_map = pickle.load(f)


def fit(w2v, ds, iters=1000, window_rad=4, lr=1e-4, batch_size=32, sample_size=32, beta=0.9):

    iterds = itertools.cycle(ds["train"])

    n_dim = w2v.n_dim

    do_batch = np.zeros((batch_size, n_dim))
    dc_batch = np.zeros((batch_size, n_dim))

    b_idx = 0

    O_batch = []
    O_words_batch = [] # list(list(str))

    c_batch = []
    c_words_batch = [] # list(str)


    momentum = {}


    for i in range(iters):
        x = next(iterds)
        no_punc = re.sub("[^\w\s]", "", x["text"])
        words = no_punc.lower().split()



        for i in range(len(words)):
            
            # Collect outer words 
            O = []
            o_words_visited = []
            for j in range(-window_rad, window_rad + 1):
                if j == 0:
                    continue
                try:
                    O.append(w2v.word_map[words[i + j]][0])
                    o_words_visited.append(words[i + j])
                except:
                    continue

            # Collect center word
            try: 
                c = w2v.word_map[words[i]][1]
                c_word_visited = words[i]
            except:
                continue

            try:
                O = np.concatenate(O, axis=0)
                O_batch.append(O)
                O_words_batch.append(o_words_visited)
                c_batch.append(c)
                c_words_batch.append(c_word_visited)

                b_idx += 1
            except:
                pass


            if b_idx >= batch_size:
                # compute gradients
                batch_grads = grad_batch(O_batch, c_batch, O_words_batch, c_words_batch, w2v.word_map, n_dim, sample_size=sample_size)
                
                for key in batch_grads:
                    if key not in momentum:
                        momentum[key] = [np.zeros((1, n_dim)), np.zeros((1, n_dim))]
                    momentum[key][0] = beta * momentum[key][0] + (1 - beta) * batch_grads[key][0]
                    momentum[key][1] = beta * momentum[key][1] + (1 - beta) * batch_grads[key][1]

                    w2v.word_map[key][0] = w2v.word_map[key][0] - lr * momentum[key][0]
                    w2v.word_map[key][1] = w2v.word_map[key][1] - lr * momentum[key][1]
                
                l = loss_batch(O_batch, c_batch, w2v.word_map, sample_size=sample_size)
                print(l)


                O_batch = []
                O_words_batch = [] # list(list(str))

                c_batch = []
                c_words_batch = [] # list(str)

                b_idx = 0
    
    return w2v


            

def loss(o, c, word_map, sample_size=32):
    all_keys = word_map.keys()
    keys = random.sample(all_keys, sample_size)

    n_o = []

    for key in keys:
        n_o.append(word_map[key][0])
    n_o = np.concatenate(n_o, axis=0)

    return -np.log(sig(np.dot(c, o.T))) - np.sum(np.log(sig(-np.dot(n_o, c.T))))


def loss_batch(O, C, word_map, sample_size=32):

    batch_size = len(O)
    l = 0
    for k in range(batch_size):
        c = C[k]
        for i in range(O[k].shape[0]):
            o = np.expand_dims(O[k][i], axis=0)
            l += loss(o, c, word_map=word_map, sample_size=sample_size)
    return l / 32


def sig(x):
    return 1 / (1 + np.exp(-x))
        
    

def grad(o, c, word_map, sample_size=32):
    all_keys = list(word_map.keys())
    keys = random.sample(all_keys, sample_size)

    n_o = []

    for key in keys:
        n_o.append(word_map[key][0])
    n_o = np.concatenate(n_o, axis=0)

    p_sim = sig(np.dot(c, o.T))

    do = -c  * (1 - p_sim)
    dc = -o * (1 - p_sim) + np.sum((1 - sig(np.dot(n_o, c.T))) * n_o, axis=0)

    # Compute the negative derivatives
    dns = []
    dn_words = []

    for i in range(n_o.shape[0]):
        n_oi = np.expand_dims(n_o[i], axis=0)
        dn_i = c * (1-np.log(sig(np.dot(n_oi, c.T))))
        dns.append(dn_i)
        dn_words.append(keys[i])

    # print(dc)

    return do, dc, dns, dn_words
    
def grad_batch(O, C, O_words, C_words, word_map, n_dim, sample_size=32):

    grads = {}
    batch_size = len(O)
    for k in range(batch_size):
        c = C[k]
        c_word = C_words[k]
        for i in range(O[k].shape[0]):
            o = np.expand_dims(O[k][i], axis=0)
            o_word = O_words[k][i]
            do, dc, dns, dn_words = grad(o, c, word_map, sample_size=sample_size)

            if o_word not in grads:
                grads[o_word] = [np.zeros((1, n_dim)), np.zeros((1, n_dim))]
            grads[o_word][0] += do / batch_size

            if c_word not in grads:
                grads[c_word] = [np.zeros((1, n_dim)), np.zeros((1, n_dim))]
            grads[c_word][1] += dc / (batch_size * O[k].shape[0])

            for j in range(len(dns)):
                if dn_words[j] not in grads:
                    grads[dn_words[j]] = [np.zeros((1, n_dim)), np.zeros((1, n_dim))]
                grads[dn_words[j]][0] += dns[j] / (batch_size)

    
    return grads








if __name__ == "__main__":
    ds = load_dataset("wikipedia", "20220301.en")

    # ds = {
    #     "train" : [
    #         {"text": "the quick brown fox jumped over the lazy dog"}, 
    #     ]
    # }

    
    iterds = iter(ds["train"])
    lines = next(iterds)

    w2v = Word2Vec(ds, n_dim=2)

    pprint(w2v.word_count, sort_dicts=False)
    # print(w2v.word_map)
    # x = input()

    w2v = fit(w2v, ds, iters=100000, window_rad=3, lr=1e-3, batch_size=32, sample_size=500)
    w2v.save("test")



    

    
        
    
    
