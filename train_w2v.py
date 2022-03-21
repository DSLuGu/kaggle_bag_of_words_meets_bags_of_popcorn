import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='bs4')

import os
import argparse
import logging

import pandas as pd

import nltk.data
from gensim.models import word2vec

from preprocess import review_to_words, review_to_sentences


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(CUR_DIR, 'dataset')


def define_argparser():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_nm', default='./models/w2v.features-300.min_words-40.context-10.pkl')
    
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
    for i, row in unlabeledTrainData.iterrows():
        sentences += review_to_sentences(
            row['review'], tokenizer, remove_stopwords=True
        )
    
    num_features = 300  # Word vector dimensionality
    min_word_count = 40 # Minimum word count
    num_workers = 4     # Number of threads to run in parallel
    context = 10        # Context window size
    downsampling = 1e-3 # Downsample setting for frequent words
    
    model = word2vec.Word2Vec(
        sentences, workers=num_workers, vector_size=num_features, 
        min_count=min_word_count, window=context, sample=downsampling, 
    )
    # If you don't plan to train the model any further, 
    # calling init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    
    # It can be helpful to create a meaningful model name 
    # and save the model for later use.
    # You can load it later using Word2Vec.load()
    model.save(config.model_nm)
    
    print(model.wv.most_similar("man"))
    print(model.wv.most_similar("queen"))
    
    return None



if __name__ == '__main__':
    
    config = define_argparser()
    main(config)
    