import matplotlib.pyplot as plt
import json
import os
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocess(object):
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')

        self.lemma = WordNetLemmatizer()

    def refine(self, json_list):
        sw = stopwords.words('english') + ['URL', 'AT_USER', 'rt', '…']
        data_list = list()
        for item in json_list:
            text = [self.lemma.lemmatize(word.lower()) for word in item['text'] if (word not in sw) and (word not in string.punctuation)]
            data_list.append(text)

        return data_list

    def file_parser(self, dir):
        json_list = list()
        data = open(dir)
        while True:
            l = data.readline()
            if not l:
                break
            json_list.append(json.loads(l))
        data.close()
        data_list = self.refine(json_list)

        return data_list

    def preprocess(self):
        assets = os.listdir('./preprocessed')
        print(assets)

        data_dict = dict()
        bag_of_words = list()

        for asset in assets:
            asset_dir = './preprocessed/' + asset
            dates = os.listdir(asset_dir)
            date_dict = dict()
            for date in dates:
                data_list = self.file_parser(asset_dir + '/' + date)
                for i in data_list:
                    bag_of_words += i
                date_dict[date] = data_list
            data_dict[asset] = date_dict
            print('appended', asset)

        freq = nltk.FreqDist(bag_of_words)
        plt.ion()
        freq.plot(50, cumulative=False)
        plt.savefig('word_frequency.png')
        plt.ioff()

        self.data_dict = data_dict
        self.bag_of_words = list(set(bag_of_words))

        self.save_data()

        # Data Example
        print(data_dict['AAPL']['2015-12-01'])

        # Bag of Words
        print(len(bag_of_words))

    def save_data(self):
        with open('obj/data_dict.pkl','wb') as f:
            pickle.dump(self.data_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        with open('obj/bag_of_words.pkl','wb') as f:
            pickle.dump(self.bag_of_words, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_data(self):
        with open('obj/data_dict.pkl', 'rb') as f:
            self.data_dict = pickle.load(f)
        f.close()
        with open('obj/bag_of_words.pkl', 'rb') as f:
            self.bag_of_words = pickle.load(f)
        f.close()