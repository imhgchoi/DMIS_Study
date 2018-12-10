import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim

class Word2Vec(object):
    def __init__(self, data_dict):
        self.sentences = self.sentence_list(data_dict)
        self.model = gensim.models.Word2Vec(self.sentences, size=200, min_count=1, workers=10)

    def sentence_list(self, data_dict):
        sentences = list()
        for asset, _ in data_dict.items() :
            for date, _ in data_dict[asset].items() :
                sentences += data_dict[asset][date]

        return sentences

    def train_model(self):
        self.model.train(self.sentences, total_examples=len(self.sentences), epochs=100)
        self.model.save('obj/word_embedding')

    def load_model(self):
        self.model = gensim.models.Word2Vec.load('obj/word_embedding')