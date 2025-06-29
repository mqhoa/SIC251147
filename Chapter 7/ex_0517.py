import re
import os
import nltk
import urllib
import bs4 as bs
import warnings
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')
# nltk.download()

# Connect to the source.
source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Machine_learning').read()

# Beautiful soup object.
soup = bs.BeautifulSoup(source,'lxml')

# Build a long string.
my_text = ""
for paragraph in soup.find_all('p'):
    my_text += paragraph.text
print(my_text)

my_text = my_text.lower()
my_text = re.sub(r'\[[0-9]*\]',' ', my_text)
my_text = re.sub(r'\W',' ', my_text)
#my_text = re.sub(r'\s+',' ',my_text)
my_text = re.sub(r'\d+',' ',my_text)
my_text = re.sub(r'\s+',' ',my_text)

my_sentences = nltk.sent_tokenize(my_text)
my_words_0=[]
for a_sentence in my_sentences:
    my_words_0 += nltk.word_tokenize(a_sentence)
my_words_0 = [a_word for a_word in my_words_0 if len(a_word)>2 ]
my_words_0 = [a_word for a_word in my_words_0 if a_word not in stopwords.words('english')]
my_words_0 = [my_words_0]    # Required by Word2Vec.
len(my_words_0[0])

my_model = Word2Vec(my_words_0, vector_size = 100, min_count=1)
my_words = my_model.wv.key_to_index
len(my_words)

# View the dense vector corresponding to 'machine'.
my_vector = my_model.wv['machine']
print("Length = " + str(my_vector.shape[0]))
print("-"*100)
print(my_vector)

my_model.wv.most_similar('learning')

my_model.wv.most_similar('artificial')

# Operation:
# global - cooling + warming = ???
my_model.wv.most_similar(positive=['machine','human'], negative= ['learning'])

# Load the file.
filename = "GoogleNews-vectors-negative300.bin"
a_model = KeyedVectors.load_word2vec_format(filename, binary=True)

# The most similar words to 'king' or 'kings'.
a_model.most_similar(['king','kings'])

# Operation: queen(queens) - woman(women) + man(men) = ???
a_model.most_similar(positive=['queen','queens','man','men'], negative= ['woman','women'])