import numpy as np
from sklearn import metrics
import pandas as pd
import os
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import skflow
from processing import getDocuments

# Data Prep
# ==================================================

# Load data
print("Loading data...")

path = 'texts/ADHD_various_half/4_word/'

textNames = sorted([os.path.join(path, fn) for fn in os.listdir(path)])

texts = sorted([os.path.join(path, fn) for fn in os.listdir(path)])
if len(texts) > 80: #removing .DS_Store
   texts = texts[1:len(texts)]

AD_train = texts[0:15]
AD_test = texts[15:40]
TD_train = texts[40:55]
TD_test = texts[55:80]

AD_trainN = textNames[0:15]
AD_testN = textNames[15:40]
TD_trainN = textNames[40:55]
TD_testN = textNames[55:80]

AD_train = getDocuments(AD_train, 'none', False, AD_trainN)
AD_test = getDocuments(AD_test, 'none', False, AD_testN)
TD_train = getDocuments(TD_train, 'none', False, TD_trainN)
TD_test = getDocuments(TD_test, 'none', False, TD_testN)


def listsfromDict(inputDict):
    newList = []
    for key, value in inputDict.iteritems():
        valList = []
        for val in value:
            valList.append(str(val))
        newList.append(valList)
    return newList

AD_train = listsfromDict(AD_train)
AD_test = listsfromDict(AD_test)
TD_train = listsfromDict(TD_train)
TD_test = listsfromDict(TD_test)

print len(AD_train + AD_test)
# Generate labels
AD_labels = [[0] for _ in AD_train + AD_test] #[0, 1]
TD_labels = [[1] for _ in TD_train + TD_test] #[1, 0]
y = np.concatenate([AD_labels, TD_labels], 0)

x = np.asarray(AD_train + AD_test + TD_train + TD_test)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
y_shuffled = y[shuffle_indices]
x_shuffled = x[shuffle_indices]

# Split train/test set
# TODO: Should use cross-validation
X_train, X_test = x_shuffled[0:40], x_shuffled[40:80]
y_train, y_test = y_shuffled[0:40], y_shuffled[40:80]
print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))

X_train = [(' ').join(l) for l in X_train]
X_test = [(' ').join(l) for l in X_test]

### Process vocabulary

MAX_DOCUMENT_LENGTH = 5

vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

### Models

EMBEDDING_SIZE = 50

def average_model(X, y):
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    features = tf.reduce_max(word_vectors, reduction_indices=1)
    return skflow.models.logistic_regression(features, y)

def rnn_model(X, y):
    """Recurrent neural network model to predict from sequence of words
    to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].

    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = rnn_cell.GRUCell(EMBEDDING_SIZE)
    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = rnn.rnn(cell, word_list, dtype=tf.float32)
    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    return skflow.models.logistic_regression(encoding, y)

classifier = skflow.TensorFlowEstimator(model_fn=rnn_model, n_classes=15,
    steps=1000, optimizer='Adam', learning_rate=0.01, continue_training=True)

# Continously train for 1000 steps & predict on test set.
while True:
    classifier.fit(X_train, y_train)
    score = metrics.accuracy_score(y_test, classifier.predict(X_test))
    print('Accuracy: {0:f}'.format(score))
