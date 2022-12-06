import json, glob, os, random
import re, tensorflow.keras, os
import pandas as pd, keras, io
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input,Dropout, Embedding, LSTM, Bidirectional, Activation, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from bpe import Encoder
from tensorflow.keras.backend import reshape
import keras.backend as K
from scipy.stats import spearmanr
from itertools import permutations
import argparse


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def load_vectors(fname, word_index):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if word_index.get(tokens[0],-1) != -1:
            data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def preprocess_one(args, context):
    tw_subtokens = re.findall(r'\w+', context.lower())
    if len(tw_subtokens) > args.max_token_sent:
        tw_subtokens = tw_subtokens[:args.max_token_sent]
    return (' '.join(tw_subtokens))

def preprocess(args, sents, labels):
    assert len(sents) == len(labels)
    output_sents = []; 
    for idx in range(len(sents)):
        conts = []
        for idy in range(args.num_sent+1):
            cont = preprocess_one(args, sents[idx][idy])
            conts.append(cont)
        output_sents.append(conts)
    return output_sents, labels

def get_accuracy(probs, gold):
    assert len(probs) == len(gold)
    idx = 0; true = 0
    while idx < len(probs):
        if probs[idx] > probs[idx+1]: # index idx must be the answer
            true+=1
        idx+=2
    return true/(len(gold)/2)

def model_with_fasttext(train_sent, dev_sent, test_sent, \
        train_denom, dev_denom, test_denom, \
        train_label, dev_label, test_label, \
        tokenizer, args):
    word_index = tokenizer.word_index
    nb_words = min(args.max_nb_words, len(word_index))
    print('Total words in dict:', nb_words)
    embeddings = load_vectors(args.fasttext_path, word_index)
    embedding_matrix = np.zeros((nb_words + 1, args.embedding_dim))
    for word, i in word_index.items():
        if i > args.max_nb_words:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(-4.2, 4.2, args.embedding_dim)
    print('Finish to read Fast Text Embedding')
    
    print('Begin the training!')
    best_rankCorr_dev = 0.0; best_rankCorr_test = 0.0;
    with tf.device('/gpu:0'):
        embedding_layer = Embedding(nb_words + 1, args.embedding_dim, weights=[embedding_matrix], 
                                input_length=args.max_token_sent, trainable=True, mask_zero=True)
        sent = Input(shape=(args.num_sent+1, args.max_token_sent,), dtype='int32', name='sentence')
        denom = Input(shape=(args.num_sent+1,), dtype='float32', name='denom')

        embedded_sent = embedding_layer(sent) #batch x 5 x #token x #hidden
        mask_sent = Masking(mask_value=0)(sent)
        mask_sent = tf.cast(mask_sent, tf.float32)

        lstm1 = LSTM(units=200, activation='tanh', recurrent_activation='hard_sigmoid', 
                recurrent_regularizer=keras.regularizers.l2(0.2), return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        bilstm1 = Bidirectional(lstm1, merge_mode='concat')
        
        batch_size, num_sent, num_token, hidden_size = embedded_sent.shape
        embedded_sent = reshape(embedded_sent, shape=(-1, num_token, hidden_size))
        
        mask_sent = reshape(mask_sent, shape=(-1, args.max_token_sent, 1)) #batch * (args.num_sent + 1) x #word x 1
        sent_vector = bilstm1(embedded_sent)  * mask_sent #batch * (args.num_sent + 1) x # word x #hidden
        # averaging after bilstm
        sent_vector = tf.keras.backend.sum(sent_vector, axis=1, keepdims=False) #batch *(args.num_sent + 1)  x #hidden
        sent_vector = sent_vector/reshape(denom, shape=(-1, 1))
        sent_vector = reshape(sent_vector, shape=(-1, args.num_sent+1, 400)) #batch x (args.num_sent + 1) x #hidden

        lstm2 = LSTM(units=200, activation='tanh', recurrent_activation='hard_sigmoid', 
                recurrent_regularizer=keras.regularizers.l2(0.2), return_sequences=False, dropout=0.3, recurrent_dropout=0.3)
        bilstm2 = Bidirectional(lstm2, merge_mode='concat')
        sent_vector2 = bilstm2(sent_vector)  #batch x #hidden
        mlp = Dense(1, activation='sigmoid')
        
        output = mlp(sent_vector2) #batch x 1
        sent_model = Model([sent, denom], [output])
        bce = tf.keras.losses.BinaryCrossentropy()

        sent_model.compile(loss=bce, optimizer='adam')

        best_acc_dev=0.0; best_acc_test=0.0; patience = 0
        for i in range(args.iterations):
            if patience == args.patience:
                break
            sent_model.fit([train_sent, train_denom], [train_label], batch_size=args.batch_size, epochs=1, shuffle=True, verbose=True)
            
            prob_distrib = sent_model.predict([dev_sent, dev_denom], batch_size=1000)
            acc_dev = get_accuracy(prob_distrib, dev_label)
            if acc_dev > best_acc_dev:
                print('Epoch ' + str(i) +' has better dev acc: '+ str(acc_dev))
                best_acc_dev = acc_dev
                patience = 0
                #Predict Test
                prob_distrib = sent_model.predict([test_sent, test_denom], batch_size=1000)
                best_acc_test = get_accuracy(prob_distrib, test_label)
                print('Epoch ' + str(i) +' test acc: '+ str(best_acc_test))
            else:
                patience += 1
    
    print('Dev Acc:', best_acc_dev,'Test Acc:', best_acc_test)
    print('-----------------------------------------------------------------------------------')
    return best_acc_dev, best_acc_test


def tokenize(tokenizer, data):
    tokenized_sents = []
    denom_sents = []
    for idx, datum in enumerate(data):
        tokenized_datum = tokenizer.texts_to_sequences(datum)
        denom_sent = [args.max_token_sent] * (args.num_sent + 1)
        for idy in range(args.num_sent+1):
            if len(tokenized_datum[idy]) != 0 and len(tokenized_datum[idy]) < args.max_token_sent:
                denom_sent[idy] = len(tokenized_datum[idy])
        tokenized_sents.append(tokenized_datum)
        denom_sents.append(denom_sent)
    return tokenized_sents, denom_sents

def train_and_test_fasttext(trainset, devset, testset, args):
    train_sent, train_label = trainset
    dev_sent, dev_label = devset
    test_sent, test_label = testset
    
    fulldata=[]
    for sents in train_sent:
        for sent in sents:
            fulldata.append(sent)
    fulldata=np.array(fulldata)
    tokenizer = Tokenizer(num_words=args.max_nb_words, lower=True)
    tokenizer.fit_on_texts(fulldata)
    
    train_sent, train_denom = tokenize(tokenizer, train_sent)
    dev_sent, dev_denom = tokenize(tokenizer, dev_sent)
    test_sent, test_denom = tokenize(tokenizer, test_sent)
    train_sent = [sequence.pad_sequences(sent, maxlen=args.max_token_sent, padding='post') for sent in train_sent]
    dev_sent = [sequence.pad_sequences(sent, maxlen=args.max_token_sent, padding='post') for sent in dev_sent]
    test_sent = [sequence.pad_sequences(sent, maxlen=args.max_token_sent, padding='post') for sent in test_sent]
    
    train_sent = np.array(train_sent)
    test_sent = np.array(test_sent)
    dev_sent = np.array(dev_sent)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    dev_label = np.array(dev_label)
    train_denom = np.array(train_denom, dtype=np.float)
    test_denom = np.array(test_denom, dtype=np.float)
    dev_denom = np.array(dev_denom, dtype=np.float)

    return model_with_fasttext(train_sent, dev_sent, test_sent, train_denom, dev_denom, test_denom, train_label, dev_label, test_label, tokenizer, args)



def read_data(fname, num_sent=4):
    contexts = []
    labels = []
    data = pd.read_csv(fname)
    for idx, row in data.iterrows():
        sents = []
        for i in [4,3,2,1]:
            if len(sents) == num_sent:
                break
            sents.insert(0, row[f'Kalimat-{i}'])
        #context = ' '.join(sents) # row['Kalimat-1'] +' '+ row['Kalimat-2'] +' '+ row['Kalimat-3'] +' '+ row['Kalimat-4']
        ending1 = row['Correct Ending']
        ending2 = row['Incorrect Ending']
        
        contexts.append(sents+[ending1])
        labels.append(1)
        
        contexts.append(sents+[ending2])
        labels.append(0)
    return contexts, labels

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--data_path', default='data/', help='path to all train/test/dev folds')
args_parser.add_argument('--max_nb_words', type=int, default=50000, help='maximum size of vocabulary')
args_parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dimension of fasttext')
args_parser.add_argument('--num_class', type=int, default=1, help='number of class, we set 1 here because we use sigmoid')
args_parser.add_argument('--patience', type=int, default=20, help='patience count for early stopping')
args_parser.add_argument('--iterations', type=int, default=100, help='total epoch')
args_parser.add_argument('--batch_size', type=int, default=20, help='total batch size')
args_parser.add_argument('--fasttext_path', default='/home/ffajri/Data/Indonesian/cc.id.300.vec', help='path to indonesian fasttext')
args_parser.add_argument('--max_token_sent', type=int, default=30, help='maximum word allowed for 1 sent')
args_parser.add_argument('--num_sent', type=int, default=4, help='number of sentence in context')
args_parser.add_argument('--seed', type=int, default=1, help='random seed')

args = args_parser.parse_args()


for num_sent in [1,2,3,4]:
    args.num_sent = num_sent
    set_seed(args.seed)
    trainset = read_data('Data/data_id/train.csv', args.num_sent)
    devset = read_data('Data/data_id/dev.csv', args.num_sent)
    testset = read_data('Data/data_id/test.csv', args.num_sent)
    train_dataset = preprocess(args, trainset[0], trainset[1])
    dev_dataset = preprocess(args, devset[0], devset[1])
    test_dataset = preprocess(args, testset[0], testset[1])

    dev_score, test_score = train_and_test_fasttext(train_dataset, dev_dataset, test_dataset, args)
    print('Num Sent:', num_sent)
    print('Dev set accuracy', dev_score)
    print('Test set accuracy', test_score)
    print('-------------------------------------------')

