import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
         'h', 'i', 'j', 'k', 'l', 'm', 'n',
         'o', 'p', 'q', 'r', 's', 't', 'u',
         'v', 'w', 'x', 'y', 'z']
MAX_WORD_LEN = 30


def encode(word):
    encoded_array = [0]*MAX_WORD_LEN*len(chars)
    counter = 0
    word.lower().replace(" ", "")
    for char in word:
        for option in chars:
            if char == option:
                encoded_array[counter + chars.index(option)] = 1
        counter += len(chars)
    return encoded_array


def get_file_codes(file, words):
    a = []
    for num in range(words):
        a.append(file.readline())
    language_a = np.empty((len(a), MAX_WORD_LEN*len(chars)))
    print("Current File: " + file.name)
    for num in range(len(a)):
        temp = np.array(encode(a[num]))
        for y in range(MAX_WORD_LEN*len(chars)):
            language_a[num, y] = temp[y]
    print("All words are now encoded.")
    return language_a


# The number of words to be trained/tested from each language
TRAIN_SIZE = 5000
TEST_SIZE = 5000
lang1 = 'English'
lang2 = 'Spanish'

if lang1 == 'English':
    with open(r'C:\Users\benfo\Downloads\english_training.txt', 'r') as f:
        train1 = get_file_codes(f, TRAIN_SIZE)
        test1 = get_file_codes(f, TEST_SIZE)
elif lang1 == 'Spanish':
    with open(r'C:\Users\benfo\Downloads\spanish_training.txt', 'r') as f:
        train1 = get_file_codes(f, TRAIN_SIZE)
        test1 = get_file_codes(f, TEST_SIZE)
else:
    with open(r'C:\Users\benfo\Downloads\german_training.txt', 'r') as f:
        train1 = get_file_codes(f, TRAIN_SIZE)
        test1 = get_file_codes(f, TEST_SIZE)


if lang2 == 'English':
    with open(r'C:\Users\benfo\Downloads\english_training.txt', 'r') as f:
        train2 = get_file_codes(f, TRAIN_SIZE)
        test2 = get_file_codes(f, TEST_SIZE)
elif lang2 == 'Spanish':
    with open(r'C:\Users\benfo\Downloads\spanish_training.txt', 'r') as f:
        train2 = get_file_codes(f, TRAIN_SIZE)
        test2 = get_file_codes(f, TEST_SIZE)
else:
    with open(r'C:\Users\benfo\Downloads\german_training.txt', 'r') as f:
        train2 = get_file_codes(f, TRAIN_SIZE)
        test2 = get_file_codes(f, TEST_SIZE)


train_inputs = np.vstack((train1, train2))
test_inputs = np.vstack((test1, test2))


train_outputs = np.empty(TRAIN_SIZE * 2)
train_outputs[:TRAIN_SIZE] = 1
train_outputs[TRAIN_SIZE:] = 0

test_outputs = np.empty(TEST_SIZE * 2)
test_outputs[:TEST_SIZE] = 1
test_outputs[TEST_SIZE:] = 0
print("Version", tf.__version__)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(780, )),
    keras.layers.Dense(70, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              )

# Create validation set
x_val = train_inputs[TRAIN_SIZE - 1000: TRAIN_SIZE + 1000]
y_val = train_outputs[TRAIN_SIZE - 1000: TRAIN_SIZE + 1000]
train_inputs = np.vstack((train_inputs[TRAIN_SIZE + 1000:], train_inputs[:TRAIN_SIZE - 1000]))
train_outputs = np.concatenate((train_outputs[TRAIN_SIZE + 1000:], train_outputs[:TRAIN_SIZE - 1000]))
# Train the model
history = model.fit(train_inputs, train_outputs, epochs=500, batch_size=500, validation_data=(x_val, y_val))

# Evaluate test inputs/outputs as a whole
test_loss, test_acc = model.evaluate(test_inputs, test_outputs)
print("Languages compared: {} and {}".format(lang1, lang2))
print('Final Accuracy(Test Inputs): {:.2f}%'.format(100 * test_acc))
print('Final Loss(Test Inputs): {:.2f}%'.format(100 * test_loss))

history_dict = history.history
acc = history_dict['acc']
loss = history_dict['loss']
val_acc = history_dict['val_acc']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure('Close this plot to test individual words in the console', figsize=(8, 6))
plt.plot(epochs, loss, '#F08080', label='Training Loss', linewidth=3)
plt.plot(epochs, val_loss, '#FF0000', label='Validation Loss', linewidth=3)
plt.plot(len(acc), test_loss, '#8B0000', label='Testing Loss', marker='x', linewidth=2)
plt.plot(epochs, acc, '#1E90FF', label='Training Accuracy', linewidth=3)
plt.plot(epochs, val_acc, '#00008B', label='Validation Accuracy', linewidth=3)
plt.plot(len(acc), test_acc, '#191970', label='Testing Accuracy', marker='x', linewidth=2)
plt.title('Loss and Accuracy: Comparing {} vs. {}'.format(lang1, lang2))
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend(loc=2, fontsize=8)
plt.show()

while True:
    word = input("Enter a word or phrase to be tested.")
    test_inputs[-1] = encode(word)
    prediction = model.predict(test_inputs)
    if prediction[-1][0] > .5:
        print("I predict {} with {:.2f}% confidence. How did I do?". format(lang2, 100 * prediction[-1][0]))
    else:
        print("I predict {} with {:.2f}% confidence. How did I do?".format(lang1, 100 * prediction[-1][1]))
    print()

