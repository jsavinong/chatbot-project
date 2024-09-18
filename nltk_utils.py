import nltk
#nltk.download('punkt')
#nltk.download('punkt_tab')
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('spanish')
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

