import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import DataLoader
from trainer import Trainer

from models.cnn import CNNClassifier
from models.rnn import RNNClassifier


def define_argparser():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required=True, type=str)
    p.add_argument('--train_fn', required=True, type=str, default='labeledTrainData.tsv')
    p.add_argument('--test_fn', type=str, default='testData.tsv')
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)
    
    p.add_argument('--min_vocab_freq', type=int, default=5)
    p.add_argument('--max_vocab_size', type=int, default=9999999)
    
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)
    
    p.add_argument('--word_vec_size', type=int, default=256)
    p.add_argument('--dropout', type=float, default=.3)
    
    p.add_argument('--max_length', type=int, default=256)

    p.add_argument('--rnn', action='store_true')
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=4)
    
    p.add_argument('--cnn', action='store_true')
    p.add_argument('--use_batch_norm', action='store_true')
    p.add_argument('--window_sizes', type=int, nargs='*', default=[3, 4, 5])
    p.add_argument('--n_filters', type=int, nargs='*', default=[100, 100, 100])
    
    config = p.parse_args()
    
    return config


def main(config):
    
    loaders = DataLoader(
        train_fn=config.train_fn, batch_size=config.batch_size, 
        min_freq=config.min_vocab_freq, max_vocab=config.max_vocab_size, 
        device=config.gpu_id
    )
    
    if config.verbose >= 2:
        print(
            "|train| =", len(loaders.train_loader.dataset), 
            "|valid| =", len(loaders.valid_loader.dataset), 
        )
    
    vocab_size = len(loaders.text.vocab)
    n_classes = len(loaders.label.vocab)
    
    if config.verbose >= 2:
        print("|vocab| =", vocab_size, "|classes| =", n_classes)
    
    if config.cnn is False and config.rnn is False:
        raise Exception("You need to specify an architecture to train. (--cnn or --rnn)")
    
    if config.cnn:
        model = CNNClassifier(
            input_size=vocab_size, word_vec_size=config.word_vec_size, 
            n_classes=n_classes, use_batch_norm=config.use_batch_norm, 
            dropout_p=config.dropout, window_sizes=config.window_sizes, 
            n_filters=config.n_filters, 
        )
        optimizer = optim.Adam(model.parameters())
        criterion = nn.NLLLoss()
        if config.verbose >= 2: print(model)
        
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            criterion.cuda(config.gpu_id)
        
        cnn_trainer = Trainer(config)
        cnn_model = cnn_trainer.train(
            model, criterion, optimizer, 
            loaders.train_loader, loaders.valid_loader, 
        )
    
    if config.rnn:
        model = RNNClassifier(
            input_size=vocab_size, word_vec_size=config.word_vec_size, 
            hidden_size=config.hidden_size, n_classes=n_classes, 
            n_layers=config.n_layers, dropout_p=config.dropout, 
        )
        optimizer = optim.Adam(model.parameters())
        criterion = nn.NLLLoss()
        if config.verbose >= 2: print(model)
        
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            criterion.cuda(config.gpu_id)
        
        rnn_trainer = Trainer(config)
        rnn_model = rnn_trainer.train(
            model, criterion, optimizer, 
            loaders.train_loader, loaders.valid_loader, 
        )
    
    torch.save({
        'cnn': cnn_model.state_dict() if config.cnn else None, 
        'rnn': rnn_model.state_dict() if config.rnn else None, 
        'config': config, 
        'vocab': loaders.text.vocab, 
        'classes': loaders.label.vocab, 
    }, config.model_fn)
    
    return None


if __name__ == '__main__':
    
    config = define_argparser()
    main(config)
    