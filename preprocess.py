import sys
import fileinput

import re
from bs4 import BeautifulSoup

import nltk
# nltk.download('stopwords')
# nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def review_to_words(review, remove_stopwords=True):
    
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessing movie review)
    
    # 1. Remove HTML
    review = BeautifulSoup(review, 'html.parser').get_text()
    
    # 2. Remove non-letters
    review = re.sub("[^a-zA-Z]", " ", review)
    
    # 3. Convert to lower case, split into individual words
    words = review.lower().split()
    
    # 4. In Python, searching a set is much faster than searching
    # a list, so convert the stop words to a set
    if remove_stopwords:
        stops = set(stopwords.words('english'))
    
        # 5. Remove stop words, 6. Lemmmatizer and stemming
        lemmatizer = WordNetLemmatizer()
        porterStemmer = PorterStemmer()
        meaningful_words = [lemmatizer.lemmatize(porterStemmer.stem(w)) for w in words if not w in stops]
    
    # 7. Join the words back into one string separated by space, 
    # and return the result.
    return " ".join(meaningful_words)

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    
    # Function to split a review into parsed sentences.
    # Returns a list of sentences, where each sentence is a list of words
    
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_words to get a list of words
            sentences += [review_to_words(raw_sentence, remove_stopwords).split()]
    
    # Return the list of sentences (each sentence is a list of words, 
    # so this returns a list of lists)
    return sentences


if __name__ == '__main__':
    
    lines = iter(fileinput.input())
    first_line = next(lines)
    sys.stdout.write(first_line)
    
    for line in lines:
        if line.strip() != "":
            if len(first_line.strip().split()) == 2:
                _id, _review = line.strip().split('\t')
                _review = review_to_words(_review)
                sys.stdout.write('\t'.join([_id, _review]) + '\n')
            else:
                _id, _sentiment, _review = line.strip().split('\t')
                _review = review_to_words(_review)
                sys.stdout.write('\t'.join([_id, _sentiment, _review]) + '\n')
        else:
            sys.stdout.write('\n')
    