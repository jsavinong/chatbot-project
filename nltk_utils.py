import nltk
#nltk.download('punkt')
#nltk.download('punkt_tab')
from nltk.stem.snowball import SnowballStemmer
import numpy as np

stemmer = SnowballStemmer('spanish')
def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organiza", "organizar", "organizando"]
    words = [stem(w) for w in words]
    -> ["organiza", "organiza", "organiza"]
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["Hola", "En", "qué", "te", "puedo", "ayudar"]
    words = ["Hola", "amigo", "qué", "puedo", "hacer", "para", "ayudarte"]
    bow   = [  1 ,    0 ,    1 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # stemmed_words = [stem(w) for w in words]  # Stem the words list as well
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        # print(w, sentence_words )
        if w in sentence_words: 
            bag[idx] = 1

    return bag

# sentence = ["Hola", "En", "qué", "te", "puedo", "ayudar"]
# words = ["Hola", "amigo", "qué", "puede", "hacer", "para", "ayudarte"]
# bow = bag_of_words(sentence, words)
# print(bow)