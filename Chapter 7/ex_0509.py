import nltk
from numpy.random import randint, seed
from sklearn.feature_extraction.text import CountVectorizer

# Text data for training.
my_text = """Machine learning is the scientific study of algorithms and statistical models that computer systems use to effectively perform a specific task without using explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task.[1][2]:2 Machine learning algorithms are used in the applications of email filtering, detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory data analysis through unsupervised learning In its application across business problems, machine learning is also referred to as predictive analytics."""

n = 3  # Can be changed to a number equal or larger than 2.
n_min = n
n_max = n
n_gram_type = 'word'  # n-Gram with words.
vectorizer = CountVectorizer(ngram_range=(n_min, n_max), analyzer=n_gram_type)

n_grams = vectorizer.fit(my_text).get_feature_names_out()  # Get the n-Grams as a list.
n_gram_cts = vectorizer.transform(my_text).toarray()  # The output is an array of array.
n_gram_cts = list(n_gram_cts[0])  # Convert into a simple list.

list(zip(n_grams, n_gram_cts))  # Make a list of tuples and show.

n = 3  # Can be changed to a number equal or larger than 2.
n_min = n
n_max = n
n_gram_type = 'word'
vectorizer = CountVectorizer(ngram_range=(n_min, n_max), analyzer=n_gram_type)

n_grams = vectorizer.fit(my_text).get_feature_names_out()  # A list of n-Grams.
my_dict = {}
for a_gram in n_grams:
    words = nltk.word_tokenize(a_gram)
    a_nm1_gram = ' '.join(words[0:n - 1])  # (n-1)-Gram.
    next_word = words[-1]  # Word after the a_nm1_gram.
    if a_nm1_gram not in my_dict.keys():
        my_dict[a_nm1_gram] = [next_word]  # a_nm1_gram is a new key. So, initialize the dictionary entry.
    else:
        my_dict[a_nm1_gram] += [next_word]  # an_nm1_gram is already in the dictionary.

# View the dictionary.
my_dict


# Helper function that picks the following word.
def predict_next(a_nm1_gram):
    value_list_size = len(my_dict[a_nm1_gram])  # length of the value corresponding to the key = a_nm1_gram.
    i_pick = randint(0, value_list_size)  # A random number from the range 0 ~ value_list_size.
    return (my_dict[a_nm1_gram][i_pick])  # Return the randomly chosen next word.


# Test.
input_str = 'order to'  # Has to be a VALID (n-1)-Gram!
predict_next(input_str)

# Another test.
# Repeat for 10 times and see that the next word is chosen randomly with a probability proportional to the occurrence.
input_str = 'machine learning'  # Has to be a VALID (n-1)-Gram!
for i in range(10):
    print(predict_next(input_str))

# Initialize the random seed.
seed(123)

# A seed string has to be input by the user.
my_seed_str = 'machine learning'  # Has to be a VALID (n-1)-Gram!
# my_seed_str = 'in order'                                         # Has to be a VALID (n-1)-Gram!

a_nm1_gram = my_seed_str
output_string = my_seed_str  # Initialize the output string.
while a_nm1_gram in my_dict:
    output_string += " " + predict_next(a_nm1_gram)
    words = nltk.word_tokenize(output_string)
    a_nm1_gram = ' '.join(words[-n + 1:])  # Update a_nm1_gram.

# Output the predicted sequence.
output_string

