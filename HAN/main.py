from preprocess import Preprocess
from word2vec import Word2Vec
from news_embedding import NewsEmbeding
from han import HAN


if __name__=='__main__' :
    PREPROCESSED = True
    EMBEDDING_TRAINED = False
    WORD_EMBEDDING_READY = False


    preprocess = Preprocess()
    if not PREPROCESSED :
        preprocess.preprocess()
    preprocess.load_data()

    word2vec = Word2Vec(preprocess.data_dict)
    if not EMBEDDING_TRAINED :
        print('training word2vec...')
        word2vec.train_model()
    word2vec.load_model()

    news_emb = NewsEmbeding(word2vec.model, preprocess.data_dict)
    if not WORD_EMBEDDING_READY :
        news_emb.embed()
    news_emb.load_embeddings()
    news_emb.get_max_corpus_date_count()
    news_emb.pad_embeddings()
    #print(news_emb.embedict_padded['AAPL'])


    HAN = HAN(news_emb.emb_node_num, news_emb.date_num)

