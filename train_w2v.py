import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='bs4')

import os
import argparse
import logging

import pandas as pd

from tqdm import tqdm

import nltk.data
from gensim.models import word2vec, KeyedVectors

from preprocess import review_to_words, review_to_sentences


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(CUR_DIR, 'dataset')


def define_argparser():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_nm', default='./models/word2vec.model')
    
    p.add_argument('--train_fn', type=str, default='unlabeledTrainData.tsv')
    
    config = p.parse_args()
    
    return config

def load_dataset(dataset_dir=DATASET_DIR, dataset_nm=None):
    
    dataset_dir = os.path.join(dataset_dir, dataset_nm)
    dataset = pd.read_csv(dataset_dir, delimiter='\t', quoting=3) # quoting=3 설정은 큰따옴표를 무시
    
    return dataset

def main(config):
    
    unlabeledTrainData = load_dataset(dataset_nm=config.train_fn)
    
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    # Preprocess review
    # unlabeledTrainData['prep_review'] = unlabeledTrainData['review'].apply(
    #                                     review_to_sentences, args=(True, ))
    sentences = []
    for i, row in tqdm(unlabeledTrainData.iterrows()):
        sentences += review_to_sentences(
            row['review'], tokenizer, remove_stopwords=True
        )
        # if i == 2: break
    
    model = word2vec.Word2Vec(sentences=sentences, min_count=1, )
    # If you don't plan to train the model any further, 
    # calling init_sims will make the model much more memory-efficient.
    # model.init_sims(replace=True)
    
    # It can be helpful to create a meaningful model name 
    # and save the model for later use.
    # You can load it later using Word2Vec.load()
    model.save(config.model_nm)
    # model.wv.save_word2vec_format('./models/eng_w2v')
    # loaded_model = KeyedVectors.load_word2vec_format('./models/eng_w2v')
    
    print(model.corpus_count)
    print(model.corpus_total_words)
    
    # print(model.wv.most_similar("watch"))
    
    return None



if __name__ == '__main__':
    
    config = define_argparser()
    main(config)
    