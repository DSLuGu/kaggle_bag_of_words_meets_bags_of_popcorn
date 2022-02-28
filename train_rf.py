import os
import joblib
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

from preprocess import review_to_words


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(CUR_DIR, 'dataset')
SEED = 42


def define_argparser():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', type=str, default='labeledTrainData.tsv')
    p.add_argument('--test_fn', type=str, default='testData.tsv')
    
    p.add_argument('--n_trees', type=int, default=100)
    
    p.add_argument('--model_name', type=str, default='rf')
    p.add_argument('--n_classes', type=int, default=2)
    
    config = p.parse_args()
    
    return config


def load_dataset(dataset_dir=DATASET_DIR, dataset_nm=None):
    
    dataset_dir = os.path.join(dataset_dir, dataset_nm)
    dataset = pd.read_csv(dataset_dir, delimiter='\t', quoting=3)
    
    return dataset


def get_model(config):
    
    model = None
    if config.model_name == "rf":
        model = RandomForestClassifier(n_estimators=config.n_trees)
    
    return model


def main(config):

    model = get_model(config)
    
    train_set = load_dataset(dataset_nm=config.train_fn)
    test_set = load_dataset(dataset_nm=config.test_fn)
    
    # Preprocess review
    train_set['prep_review'] = train_set['review'].apply(review_to_words)
    test_set['prep_review'] = test_set['review'].apply(review_to_words)
        
    vectorizer = CountVectorizer(
        analyzer='word', tokenizer=None, preprocessor=None, 
        stop_words=None, max_features=5000
    )
    
    train_data_features = vectorizer.fit_transform(train_set['prep_review'])
    train_data_features = train_data_features.toarray()
    
    vocab = vectorizer.get_feature_names_out()
    
    # Train
    model.fit(train_data_features, train_set['sentiment'])
    
    # Test
    test_data_features = vectorizer.transform(test_set['prep_review'])
    test_data_features = test_data_features.toarray()
    predicted = model.predict(test_data_features)
    
    # Save result and model
    output = pd.DataFrame(data={'id': test_set['id'], 'sentiment': predicted})
    output.to_csv(
        f'{config.model_name}.{config.n_trees}.csv', 
        index=False, quoting=3
    )
    joblib.dump({'vectorizer': vectorizer, 'model': model}, config.model_fn)
    
    return None


if __name__ == '__main__':
    
    config = define_argparser()
    main(config)
    