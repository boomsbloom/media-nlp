import numpy as np
from sklearn import metrics
import pandas as pd
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from nltk.util import ngrams
import skflow, math, os
from processing import getDocuments

# Data Prep
# ==================================================

# Load data
print("Loading data...")

# path = 'texts/ADHD_various_half/4_word/'
# paths = ['texts/multiple_sites_half/NYU/AD_4',
#         'texts/multiple_sites_half/NYU/TD_4',
#         'texts/multiple_sites_half/OHSU/AD_4',
#         'texts/multiple_sites_half/OHSU/TD_4',
#         'texts/multiple_sites_half/PKU/AD_4',
#         'texts/multiple_sites_half/PKU/TD_4']

paths = ['texts/multiple_sites_full_2letter/NYU/AD_2',
        'texts/multiple_sites_full_2letter/NYU/TD_2',
        'texts/multiple_sites_full_2letter/OHSU/AD_2',
        'texts/multiple_sites_full_2letter/OHSU/TD_2',
        'texts/multiple_sites_full_2letter/PKU/AD_2',
        'texts/multiple_sites_full_2letter/PKU/TD_2']


#train_path = ['texts/multiple_sites_half/NYU/AD_4','texts/multiple_sites_half/NYU/TD_4']
train_path = ['texts/multiple_sites_full_2letter/PKU/AD_2','texts/multiple_sites_full_2letter/PKU/TD_2','texts/multiple_sites_full_2letter/NYU/AD_2','texts/multiple_sites_full_2letter/NYU/TD_2']

test_path = ['texts/multiple_sites_full_2letter/OHSU/AD_2','texts/multiple_sites_full_2letter/OHSU/TD_2']

def listsfromDict(inputDict):
    newList = []
    for key, value in inputDict.iteritems():
        valList = []
        for val in value:
            valList.append(str(val))
        newList.append(valList)
    return newList

AD_train = []
TD_train = []
AD_test = []
TD_test = []
for path in paths:

    textNames = sorted([os.path.join(path, fn) for fn in os.listdir(path)])

    texts = sorted([os.path.join(path, fn) for fn in os.listdir(path)])
    for text in texts: #removing .DS_Store
        if '.DS_Store' in text:
            texts.remove(text)

    docs = getDocuments(texts,'none',False,textNames)
    doc_list = listsfromDict(docs)

    bigram_docs = []
    for doc in doc_list:
        bigramList = []
        for item in ngrams(doc,2):
            bigramList.append("_".join(item))
        bigram_docs.append(bigramList)
    doc_list = bigram_docs

    if 'AD_' in path:
        if path in train_path:
            AD_train.append(doc_list)
        elif path in test_path:
            AD_test.append(doc_list)
    elif 'TD_' in path:
        if path in train_path:
            TD_train.append(doc_list)
        elif path in test_path:
            TD_test.append(doc_list)

AD_train = sum(AD_train,[])
AD_test = sum(AD_test,[])
TD_train = sum(TD_train,[])
TD_test = sum(TD_test,[])

print len(AD_train), len(AD_test), len(TD_train), len(TD_test)
#print len(AD_train + AD_test)
# Generate labels
#AD_labels = [[0] for _ in AD_train + AD_test] #[0, 1]
#TD_labels = [[1] for _ in TD_train + TD_test] #[1, 0]
#y = np.concatenate([AD_labels, TD_labels], 0)

#x = np.asarray(AD_train + AD_test + TD_train + TD_test)

# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# y_shuffled = y[shuffle_indices]
# x_shuffled = x[shuffle_indices]
#
# # Split train/test set
# # TODO: Should use cross-validation
# X_train, X_test = x_shuffled[0:40], x_shuffled[40:80]
# y_train, y_test = y_shuffled[0:40], y_shuffled[40:80]
# print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))

X_train = np.asarray(AD_train + TD_train)
X_test = np.asarray(AD_test + TD_test)

AD_train_labels = [[0] for _ in AD_train] #[0, 1]
TD_train_labels = [[1] for _ in TD_train] #[1, 0]
AD_test_labels = [[0] for _ in AD_test] #[0, 1]
TD_test_labels = [[1] for _ in TD_test] #[1, 0]

X_train = [(' ').join(l) for l in X_train]
X_test = [(' ').join(l) for l in X_test]

y_train = np.concatenate([AD_train_labels, TD_train_labels], 0)
y_test = np.concatenate([AD_test_labels, TD_test_labels], 0)

### Process vocabulary

#MAX_DOCUMENT_LENGTH = 5
MAX_DOCUMENT_LENGTH = 100

vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

# char_processor = skflow.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
# X_train = np.array(list(char_processor.fit_transform(X_train)))
# X_test = np.array(list(char_processor.transform(X_test)))


### Models ###

#EMBEDDING_SIZE = 50

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

EMBEDDING_SIZE = 20
N_FILTERS = 10
WINDOW_SIZE = 20
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2

def cnn_model(X, y):
    """2 layer Convolutional network to predict from sequence of words
    to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_Layer1'):
        # Apply Convolution filtering on input sequence.
        conv1 = skflow.ops.conv2d(word_vectors, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # Add a RELU for non linearity.
        conv1 = tf.nn.relu(conv1)
        # Max pooling across output of Convlution+Relu.
        pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1],
            strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
        # Transpose matrix so that n_filters from convolution becomes width.
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # Second level of convolution filtering.
        conv2 = skflow.ops.conv2d(pool1, N_FILTERS, FILTER_SHAPE2,
            padding='VALID')
        # Max across each filter to get useful features for classification.
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

    # Apply regular WX + B and classification.
    return skflow.models.logistic_regression(pool2, y)


# N_FILTERS = 10
# FILTER_SHAPE1 = [20, 256]
# FILTER_SHAPE2 = [20, N_FILTERS]
# POOLING_WINDOW = 4
# POOLING_STRIDE = 2

def char_cnn_model(X, y):
    """Character level convolutional neural network model to predict classes."""
    byte_list = tf.reshape(skflow.ops.one_hot_matrix(X, 256),
        [-1, MAX_DOCUMENT_LENGTH, 256, 1])
    with tf.variable_scope('CNN_Layer1'):
        # Apply Convolution filtering on input sequence.
        conv1 = skflow.ops.conv2d(byte_list, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # Add a RELU for non linearity.
        conv1 = tf.nn.relu(conv1)
        # Max pooling across output of Convlution+Relu.
        pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1],
            strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
        # Transpose matrix so that n_filters from convolution becomes width.
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # Second level of convolution filtering.
        conv2 = skflow.ops.conv2d(pool1, N_FILTERS, FILTER_SHAPE2,
            padding='VALID')
        # Max across each filter to get useful features for classification.
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
    # Apply regular WX + B and classification.
    return skflow.models.logistic_regression(pool2, y)


classifier = skflow.TensorFlowEstimator(model_fn=cnn_model, n_classes=15,
   steps=1000, optimizer='Adam', learning_rate=0.01, continue_training=True)

#Continously train for 1000 steps & predict on test set.
while True:
    classifier.fit(X_train, y_train)
    score = metrics.accuracy_score(y_test, classifier.predict(X_test))
    print('Accuracy: {0:f}'.format(score))
