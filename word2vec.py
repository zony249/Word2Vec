
from curses.ascii import isalpha
import numpy as np
from datasets import load_dataset
import re
from tqdm import tqdm
from pprint import pprint




class Word2Vec:
    def __init__(self, ds):
        
        self.word_count = self.count_words(ds)
        self.word_count = self.filter_top(self.word_count, num=100000)
        self.word_map = self.init_vec(self.word_count)
    
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
    def init_vec(self, word_count):
        vecs = {}
        for key in tqdm(word_count, "Generating Vectors"):
            vecs[key] = (np.random.rand(1, 256), np.random.rand(1, 256))
        return vecs





if __name__ == "__main__":
    ds = load_dataset("wikipedia", "20220301.en")
    iterds = iter(ds["train"])
    lines = next(iterds)

    w2v = Word2Vec(ds)

    pprint(w2v.word_count, sort_dicts=False)
    # print(w2v.word_map)
    # x = input()

    

    
        
    
    
