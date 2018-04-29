
import spacy
from spacy import *
import en_core_web_sm
# import textacy
# from textacy import preprocess
from en_core_web_sm import *
#import plac
import collections
import random
import os.path
import json
from azure.datalake.store import core, lib, multithread

import pathlib
import cytoolz
import numpy
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, Concatenate, Merge, Input
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import _pickle as pickle
import h5py
from time import time


# tenant_id='ce78e7b0-e32a-4428-9390-1c2ec8a59c47'
# username='valerie.hu@barings.onmicrosoft.com'
# password='avocado123A'
# token = lib.auth(tenant_id, username, password)
# adl = core.AzureDLFileSystem(token, store_name='bds')


################## Datalake store path
# model_dir = 'trainingdata_valerie/news_entities_trainingSentiment'
# train_dir = 'trainingdata_valerie/news_entities_trainingSentiment/train/'
# dev_dir = 'trainingdata_valerie/news_entities_trainingSentiment/test/'

################# Local path
model_dir = pathlib.Path('/Users/huwensi/Desktop/News model/data_news/entity_source_date')
train_dir= pathlib.Path('/Users/huwensi/Desktop/News model/data_news/entity_source_date/train')
dev_dir=pathlib.Path('/Users/huwensi/Desktop/News model/data_news/entity_source_date/test')


class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, max_length=400):
        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)

    def __init__(self, model, max_length=400):
        self._model = model
        self.max_length = max_length

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=300, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
            Xs = get_features(sentences, self.max_length)
            ys = self._model.predict(Xs)
            for sent, label in zip(sentences, ys):
                sent.doc.sentiment += label - 0.5
            for doc in minibatch:
                yield doc

    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[0])
        # Sentiment has a native slot for    a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['my_data'] = y


def read_data(data_dir, limit=0):

    examples = []

    for subdir, label in (('pos', 1), ('neg', 0)):
        ######### If data is on local we need to change edit the path:
        for filename in (data_dir / subdir).iterdir():
            with open(pathlib.Path(filename)) as file_:

        # for filename in adl.walk(data_dir + '/' + subdir):
        #      with adl.open(filename, 'rb') as file_:

                data = json.load(file_)
                #data = file_.read()  # Here will require string like input instead of bytes

        # for filename in adl.walk(data_dir + '/' + subdir):
        #     with adl.open(filename, 'r') as file_:
        #         data = file_.read()

                text=data['text']
                #source=data['source_domain']
                entities=data['entities']
                #date=data['date_publish']

                examples.append((text, label, entities))
                #examples.append([text, label, source, date, entities])

    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]

    return zip(*examples)  #turn examples into 'class tuple', length = number of observations in train/test set
    #return examples


def get_entity_feature(examples):

    # Here we create list of entities
    entities=[]
    for row in examples:
        entities.append(row)

    person_list=[]
    location_list=[]
    miscellaneous_list=[]

    for row in entities:

        per_list = []
        loc_list = []
        misc_list = []

        for i in row:

            if i['entity_label']=='PER':
                per_list.append(i['entity_text'])
            elif i['entity_label']=='LOC':
                loc_list.append(i['entity_text'])
            elif i['entity_label']=='MISC':
                misc_list.append(i['entity_text'])
            else:
                pass

        person=','.join(per_list)
        location=','.join(loc_list)
        miscellaneous=','.join(misc_list)

        person_list.append(person)
        location_list.append(location)
        miscellaneous_list.append(miscellaneous)

    return miscellaneous_list


######Add new features to text data
def text_add_feature(text, label, source, date, person_list, location_list, miscellaneous_list):
    #input should all be list

    new_examples=[]

    for i in range(len(text)):
        text=text[i]
        label=label[i]
        source=source[i]
        date=date[i]
        person=person_list[i]
        location=location_list[i]
        misc=miscellaneous_list[i]

        new_examples.append((text, label, source, date, person_list, location_list, miscellaneous_list))

    return zip(*new_examples)


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')


#########Feature hashing, the output Xs should be a numpy.ndarray of i rows and j elements for each row
#########where i is the number of sentence j is the number of tokens(max_length) in each sentence
#########Each row shoould looks like [token1_rank, token2_rank, token3_rank,.... token j_rank] just like imdb dataset
def get_features(docs, max_length):
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            if token.has_vector and not token.is_punct and not token.is_space and not token.is_stop:
                Xs[i, j] = token.rank + 1
                j += 1
                if j >= max_length:
                    break
    print(Xs, type(Xs),len(Xs))
    return Xs   #array of array of integer of length=max_length


def compile_lstm(embeddings, shape, settings):

    model1 = Sequential()
    model1.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['text_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model1.add(TimeDistributed(Dense(shape['nr_hidden'], bias=False)))


    model2=Sequential()
    model2.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['text_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model2.add(Dense(shape['nr_hidden'],activation='sigmoid'))

    # input_text = Input(shape=(None, text_length,shape['nr_hidden']))
    # input_entity = Input(shape=(None, entity_length ,shape['nr_hidden']))

    model = Sequential()
    #model.add(Dense(64, input_shape=(430,), activation='sigmoid'))
    #model.add(merge([model1,model2], mode='concat'))
    #model.add(Concatenate([input_text, input_entity]))
    model.add(Merge([model1, model2], mode='concat'))

    model.add(TimeDistributed(Dense(shape['nr_hidden'], bias=False)))
    model.add(Bidirectional(LSTM(shape['nr_hidden'], dropout_U=settings['dropout'],
                                 dropout_W=settings['dropout'])))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))

    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_embeddings(vocab):
    max_rank = max(lex.rank + 1 for lex in vocab if lex.has_vector)
    vectors = numpy.ndarray((max_rank + 1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank + 1] = lex.vector
    print(vectors,type(vectors),type(vectors[0]),vectors.shape,vectors.ndim)
    return vectors


def train(train_texts, train_entity, train_labels, dev_texts, dev_entity, dev_labels,
          lstm_shape, lstm_settings, lstm_optimizer, batch_size=100, nb_epoch=5,
          by_sentence=False):
    print("Loading spaCy")
    #nlp = spacy.load('en', entity=False)
    #nlp = spacy.load('en_core_web_sm')
    nlp = en_core_web_sm.load()
    embeddings = get_embeddings(nlp.vocab)   #using spacy en vocab
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)

    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts, batch_size=300, n_threads=3))
    dev_docs = list(nlp.pipe(dev_texts, batch_size=300, n_threads=3))
    train_entity = list(nlp.pipe(train_entity, batch_size=300, n_threads=3))
    dev_entity = list(nlp.pipe(dev_entity, batch_size=300, n_threads=3))

    print('train_docs:', type(train_docs), len(train_docs))

    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)

    train_X = get_features(train_docs, lstm_shape['text_length'])
    dev_X = get_features(dev_docs, lstm_shape['text_length'])

    train_X_add = get_features(train_entity, lstm_shape['text_length'])
    dev_X_add = get_features(dev_entity, lstm_shape['text_length'])

    model.fit([train_X,train_X_add],train_labels, validation_data=([dev_X,dev_X_add], dev_labels),
              nb_epoch=nb_epoch, batch_size=batch_size)
    return model


def evaluate(model_dir, texts, labels, max_length=400):
    def create_pipeline(nlp):
        '''
        This could be a lambda, but named functions are easier to read in Python.
        '''
        return [nlp.tagger, nlp.parser, SentimentAnalyser.load(model_dir, nlp,
                                                               max_length=max_length)]

    #nlp = spacy.load('en')
    nlp = spacy.load('en_core_web_sm')
    nlp.pipeline = create_pipeline(nlp)

    correct = 0
    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
        correct += bool(doc.sentiment >= 0.5) == bool(labels[i])
        i += 1
    return float(correct) / i



# @plac.annotations(
#     train_dir=("Location of training file or directory"),
#     dev_dir=("Location of development file or directory"),
#     model_dir=("Location of output model directory",),
#     is_runtime=("Demonstrate run-time usage", "flag", "r", bool),
#     nr_hidden=("Number of hidden units", "option", "H", int),
#     max_length=("Maximum sentence length", "option", "L", int),
#     dropout=("Dropout", "option", "d", float),
#     learn_rate=("Learn rate", "option", "e", float),
#     nb_epoch=("Number of training epochs", "option", "i", int),
#     batch_size=("Size of minibatches for training LSTM", "option", "b", int),
#     nr_examples=("Limit to N examples", "option", "n", int))


is_runtime=False
nr_hidden=64  #Try more
text_length=1000  # Shape
entity_length=30
dropout=0.5
learn_rate=0.001  # General NN config, try 0.01, 0.05, 0.1
nb_epoch=5    #Try 5
batch_size=16   #Try 32
nr_examples=-1  # Training params

max_length=text_length + entity_length

if is_runtime:
    dev_texts, dev_labels, dev_source, dev_date, dev_entities = read_data(dev_dir)
    #acc = evaluate(model_dir, dev_texts, dev_entities, dev_labels, max_length=max_length)
    #print(acc)
else:
    print("Read data")
    #train_texts, train_labels, train_source, train_date, train_entities = read_data(train_dir, limit=nr_examples)
    train_texts, train_labels, train_entities = read_data(train_dir, limit=nr_examples)
    #print(type(train_texts),len(train_texts))  #class tuple, length= number of observations
    #dev_texts, dev_labels, dev_source, dev_date, dev_entities = read_data(dev_dir, limit=nr_examples)
    dev_texts, dev_labels, dev_entities = read_data(dev_dir, limit=nr_examples)

    #Additional features data should also be tuple, length=number of observations
    train_entity=get_entity_feature(train_entities)
    dev_entity=get_entity_feature(dev_entities)

    train_labels = numpy.asarray(train_labels, dtype='int32')
    dev_labels = numpy.asarray(dev_labels, dtype='int32')

    t0 = time()
    lstm = train(train_texts, train_entity, train_labels, dev_texts, dev_entity, dev_labels,
                 {'nr_hidden': nr_hidden, 'text_length': text_length, 'entity_length':entity_length,
                  'max_length':max_length, 'nr_class': 1},  #lstm shape
                 {'dropout': dropout, 'lr': learn_rate},
                 {},
                 nb_epoch=nb_epoch, batch_size=batch_size)

    weights = lstm.get_weights()
    train_time=time()-t0

    # for weight in weights[1:]:
    #     print(weight)

    # with (model_dir / 'model').open('wb') as file_:
    #     pickle.dump(weights[1:], file_)

    ############Save model to HDF5
    #lstm.save_weights( model_dir / 'model_h5')

    print(type(lstm))

    print(lstm.to_json())

    print("train time: %0.3fs" % train_time)

    ### This will require a byte input when the file is open as binary
    with (model_dir / 'config.json').open('wb') as file_:
        file_.write(bytes(lstm.to_json(),encoding='utf-8'))

    ### Alternatively we can open the file in text mode and write str to it
    # with (model_dir / 'config.json').open('w') as file_:
    #     file_.write(lstm.to_json())

    print('Model saved to model directory')


#
# import datetime
# from dateutil import parser
#
# def get_time_feature(date):
#     #date is a tuple, length=train set
#     for row in date:






