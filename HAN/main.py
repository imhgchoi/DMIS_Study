from preprocess import Preprocess
from word2vec import Word2Vec
from news_embedding import NewsEmbeding

if __name__=='__main__' :
    PREPROCESSED = True
    EMBEDDING_TRAINED = True
    WORD_EMBEDDING_READY = True


    preprocess = Preprocess()
    if not PREPROCESSED :
        preprocess.preprocess()
    preprocess.load_data()

    word2vec = Word2Vec(preprocess.data_dict)
    if not EMBEDDING_TRAINED :
        word2vec.train_model()
    word2vec.load_model()

    news_emb = NewsEmbeding(word2vec.model, preprocess.data_dict)
    if not WORD_EMBEDDING_READY :
        news_emb.embed()
    news_emb.load_embeddings()
    print(news_emb.embedict['AAPL']['2015-12-01'])