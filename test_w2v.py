import os
import argparse

from gensim.models import Word2Vec


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(CUR_DIR, 'dataset')


def define_argparser():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_nm', default='./models/word2vec.model')
    
    config = p.parse_args()
    
    return config

def main(config):
    
    model = Word2Vec.load(config.model_nm)
    
    print(model.wv.most_similar('watch'))
    print(model.wv.most_similar('man'))
    
    return None


if __name__ == '__main__':
    
    config = define_argparser()
    main(config)
    