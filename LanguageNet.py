# A simple langauge network, which classifies candidate "words" into English, Spanish, or German.

import numpy as np
import math


def encode(word):
    encoded_array = np.zeros((780, 1))
    counter = 0
    word = word.lower().replace(" ", "")
    for char in word:
        encoded_array[counter + int(char) - 96] = 1 # use one-hot encoding on the characters
        counter += 26
    return encoded_array


def get_file_codes(file):
    a = []
    for num in range(10000):
        a.append(file.readline())
    language_a = np.empty((len(a), 780))
    print("Current File: " + file.name)
    for num in range(len(a)):
        temp = np.array(encode(a[num]))
        for y in range(780):
            language_a[num, y] = temp[y]
        if num % 2500 == 0:
            print(str(num) + " words have been encoded.")
    print("All words are now encoded.")
    return language_a


np.set_printoptions(threshold=100000)


class WordNeuralNet:
    def __init__(self):
        np.random.seed(5)
        self.synapse_weights = 2 * np.random.random((780, 1)) - 1

    @staticmethod
    def sigmoid(x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid(output, True))
            self.synapse_weights += adjustments
            if i % int(math.sqrt(training_iterations)) == 0 :
                print(str(i) + " training iterations completed")

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synapse_weights))
        return output


# First language is one, second language is zero
def two_language_analysis(one, two, iterations):
    all_words = np.vstack((one, two))
    trainingSize = len(all_words)
    print("Statistics - Number of training cycles per language comparisons: {0}  Number of words in the training data set: {1}".format(iterations, trainingSize))
    word_net = WordNeuralNet()
    outputs = np.empty((trainingSize, 1))
    outputs[:trainingSize//2] = 1
    outputs[trainingSize//2:] = 0

    word_net.train(all_words, outputs, iterations)
    return word_net.synapse_weights


def test_word(word, weights):
    encoded = encode(word)
    sum = 0
    for x in range(len(weights)):
        sum += weights[x] * encoded[x]
    return sigmoid(sum)


def sigmoid(num):
    return 1 / (1 + np.exp(-num))


# THIS IS WHAT YOU SHOULD CHANGE
max_count = 50000  # Maximum data size
training_iter = 250  # Training iterations
lang1 = "English"  # First Testing Language
lang2 = "Spanish"  # Second Testing Language
# THIS IS THE END OF WHAT YOU SHOULD CHANGE

# Next block - retrieve the 100,000 words(50,000 from each language) to be used for testing
testing_words = []
lang_words = []
file1 = None
file2 = None
if lang1 == "Spanish":
    with open(r'C:\Users\benfo\Downloads\spanish_training.txt', 'r') as f:
        file1 = get_file_codes(f)
        for words in range(max_count):
            lang_words.append(f.readline())
        testing_words = testing_words + lang_words
        f.close()
elif lang1 == "English":
    with open(r'C:\Users\benfo\Downloads\english_training.txt', 'r') as f:
        file1 = get_file_codes(f)
        for words in range(max_count):
            lang_words.append(f.readline())
        testing_words = testing_words + lang_words
        f.close()
elif lang1 == "German":
    with open(r'C:\Users\benfo\Downloads\german_training.txt', 'r') as f:
        file1 = get_file_codes(f)
        for words in range(max_count):
            lang_words.append(f.readline())
        testing_words = testing_words + lang_words
        f.close()

lang_words.clear()

if lang2 == "Spanish":
    with open(r'C:\Users\benfo\Downloads\spanish_training.txt', 'r') as f:
        file2 = get_file_codes(f)
        for words in range(max_count):
            lang_words.append(f.readline())
        testing_words = testing_words + lang_words
        f.close()
elif lang2 == "English":
    with open(r'C:\Users\benfo\Downloads\english_training.txt', 'r') as f:
        file2 = get_file_codes(f)
        for words in range(max_count):
            lang_words.append(f.readline())
        testing_words = testing_words + lang_words
        f.close()
elif lang2 == "German":
    with open(r'C:\Users\benfo\Downloads\german_training.txt', 'r') as f:
        file2 = get_file_codes(f)
        for words in range(max_count):
            lang_words.append(f.readline())
        testing_words = testing_words + lang_words
        f.close()

weights = two_language_analysis(file1, file2, training_iter)
""" Execution of mass testing/statistic recording
print("Language 1: {} ~ Language 2 {}".format(lang1, lang2))
count = 0
correct_count = 0
testNet = WordNeuralNet()
data_len = len(testing_words)
for x in range(data_len):
    result = test_word(testing_words[x], weights)
    if x == 0:
        print("Total words about to be tested: " + str(len(testing_words)))
    if x < data_len/2:  # i.e. a word of language 1 is being tested
        if result > .5:
            correct_count += 1
    else:  # i.e. a word of language 2 is being tested
        if result < .5:
            correct_count += 1
    count += 1
    if x % 2000 == 0:
        print("Words tested: {} Percentage: {:.4f} ".format(x, correct_count/count)
    if x == len(testing_words) - 1:
        print("Final Percentage: {:.4f}".format(correct_count/count))
"""
while True:
    word = input("Enter a word(or any string of characters) and the program will determine if it is {} or {}. ".format(lang1, lang2))
    result = float(test_word(word, weights))
    if result > .5:
        print("The predicted language is {} with {:.2f} percent confidence. ".format(lang1, 100 * result), end='')
    else:
        print("The predicted language is {} with {:.2f} percent confidence. ".format(lang2, 100 * (1-result)), end='')
    print("Was I right?\n")
