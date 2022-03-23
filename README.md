# kaggle_bag_of_words_meets_bags_of_popcorn [\[Link\]](https://www.kaggle.com/c/word2vec-nlp-tutorial/overview)

## Preprocessing

Train

```bash
cat ./dataset/labeledTrainData.tsv | python ./preprocess.py > ./dataset/labeledTrainData.prep.tsv
```

Test

```bash
cat ./dataset/testData.tsv | python ./preprocess.py > ./dataset/testData.prep.tsv
```

## Statistical Model

### RandomForest

Run

```bash
python train_rf.py --model_fn ./models/model.rf.n_trees-100.pkl
```

- Model: `--model_fn`

## Neural Network Model

```bash
python train.py --model_fn ./models/model.cnn.pkl --train_fn ./dataset/labeledTrainData.prep.tsv --test_fn ./dataset/testData.prep.tsv --cnn --verbose 2 --n_epochs 3
```

- Model: `--model_fn`
- Train-set: `--train_fn`
- Test-set: `--train_fn`
- CNN: `--cnn`
- RNN: `--rnn`

### Classification

```bash
cat ./dataset/testData.prep.tsv | python classify.py --model_fn ./models/model.cnn.pkl --drop_rnn > ./cnn.csv
```