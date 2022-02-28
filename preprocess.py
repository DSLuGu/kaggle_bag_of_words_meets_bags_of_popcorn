import re
from bs4 import BeautifulSoup

import nltk
# nltk.download('stopwords')
# nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def review_to_words(review):
    
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
    stops = set(stopwords.words('english'))
    
    # 5. Remove stop words, 6. Lemmmatizer and stemming
    lemmatizer = WordNetLemmatizer()
    porterStemmer = PorterStemmer()
    meaningful_words = [lemmatizer.lemmatize(porterStemmer.stem(w)) for w in words if not w in stops]
    
    # 7. Join the words back into one string separated by space, 
    # and return the result.
    return " ".join(meaningful_words)

