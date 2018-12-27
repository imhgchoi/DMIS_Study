import numpy as np
import pickle

class NewsEmbeding(object) :
    def __init__(self, model, data_dict):
        self.word2vec = model
        self.data_dict = data_dict
        self.embedict = dict()

        self.emb_node_num = None
        self.date_num = None
        self.embedict_padded = None

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

    def get_max_corpus_date_count(self):
        max_corpus_len = 0
        max_date_len = 0
        for asset, _ in self.embedict.items():
            date_counter = 0
            for date, _ in self.embedict[asset].items():
                max_corpus_len = max(max_corpus_len, len(self.embedict[asset][date]))
                date_counter += 1
            max_date_len = max(max_date_len, date_counter)
        self.emb_node_num = max_corpus_len
        self.date_num = max_date_len

    def pad_embeddings(self):
        embedict_padded = dict()
        for asset, _ in self.embedict.items():
            news_list = list()
            date_counter = 0
            for date, _ in self.embedict[asset].items():
                padding_num = self.emb_node_num - len(self.embedict[asset][date])
                news_list.append(self.embedict[asset][date] + [np.zeros(50)] * padding_num)
                date_counter += 1
            date_padded_news_list = [[np.zeros(50)] * self.emb_node_num] * (self.date_num - date_counter) + news_list
            embedict_padded[asset] = date_padded_news_list
            print('padded',asset)
        self.embedict_padded = embedict_padded

    def reshape_data(self, date_padded_news_list):
        # transpose 하는 방법 다시 보기!!
        # 일단 보류 안 쓰게 될듯
        return np.transpose(np.array(date_padded_news_list))

    def save_embeddings(self):
        with open('obj/embeddings.pkl','wb') as f:
            pickle.dump(self.embedict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_embeddings(self):
        with open('obj/embeddings.pkl', 'rb') as f:
            self.embedict = pickle.load(f)
        f.close()

