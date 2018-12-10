import numpy as np
import pickle

class NewsEmbeding(object) :
    def __init__(self, model, data_dict):
        self.word2vec = model
        self.data_dict = data_dict
        self.embedict = dict()

    def embed(self):
        for asset, _ in self.data_dict.items() :
            news_dict = dict()
            for date, _ in self.data_dict[asset].items() :
                news_list = list()
                for l in self.data_dict[asset][date] :
                    if len(l) ==0 :
                        continue
                    news_list.append(np.mean(self.word2vec[l], axis=0))
                news_dict[date] = news_list
            self.embedict[asset] = news_dict
            print('embedded',asset)
        self.save_embeddings()

    def save_embeddings(self):
        with open('obj/embeddings.pkl','wb') as f:
            pickle.dump(self.embedict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_embeddings(self):
        with open('obj/embeddings.pkl', 'rb') as f:
            self.embedict = pickle.load(f)
        f.close()

