# -*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    mean_absolute_error, mean_squared_error, median_absolute_error
from classification_utils import SequenceClassificationModel

from torch_utils import print_model

import pandas as pd
from os import listdir


# Lecture des fichiers
def splitting(text):
    # print(text)
    only_text = text.split(",", 1)[1]
    pol = text.split(",", 1)[0]
    return pol, only_text


tweets_files = listdir("tweets_data/")
df_list = []

for file in tweets_files:
    print(file)
    df = pd.read_csv('tweets_data/'+file, sep='ùù')
    df['polarity'], df['text'] = zip(*df.iloc[:, 0].apply(splitting))
    df = df[['polarity', 'text']]

    print(df.shape)
    df_list.append(df)

all_tweets = pd.concat(df_list)

all_tweets['polarity'] = all_tweets['polarity'].astype('int')
all_tweets['polarity'] = all_tweets['polarity'].replace(4, 1)

# Reduce number of tweets for quicker results
all_tweets = pd.concat([all_tweets[all_tweets['polarity'] == 0].iloc[:1000],
                        all_tweets[all_tweets['polarity'] == 1].iloc[:1000]])

print(all_tweets.shape)
print(all_tweets.describe())

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Train & eval
train = True
evaluate = False
predict = False

freeze_bert_encoder = True

model_type = 'camembert'
model_name = 'camembert-base'

model_args = {'output_dir': 'output_model_1/', 'reprocess_input_data': True,
              'overwrite_output_dir': True, 'num_train_epochs': 2, 'learning_rate': 8e-5,
              'logging_steps': 50, 'evaluate_during_training': True, 'save_steps': 2000}


# Selecting columns and rows
processed_df = all_tweets

test_size = 0.1
# stratifier = IterativeStratification(n_splits=2, order=2,
#                                      sample_distribution_per_fold=[test_size, 1.0-test_size],
#                                      random_state=0)

stratifier = StratifiedShuffleSplit(n_splits=2, random_state=0, test_size=test_size)

# train_indexes, test_indexes = next(stratifier.split(processed_df, processed_df.drop('text', axis=1)))
train_indexes, test_indexes = next(stratifier.split(processed_df, processed_df['polarity']))

train_df = processed_df.iloc[train_indexes]
eval_df = processed_df.iloc[test_indexes]

num_labels = processed_df.drop('text', axis=1).max().astype(int).tolist()
num_labels = [num_lab + 1 for num_lab in num_labels]


label_categories = ['polarity']

eval_df_unzip = eval_df

if label_categories[0] != 'labels':
    train_df.rename({label_categories[0]: 'labels'}, axis=1, inplace=True)
    eval_df.rename({label_categories[0]: 'labels'}, axis=1, inplace=True)


print(train_df.head())
print(eval_df.head())

print('\n# Number of categories: %s\n# Number of labels: %s' % (len(num_labels), num_labels))

num_labels = num_labels[0]
print('\n# Using mono-output classification model: %s from %s' % (model_type, model_name))
model = SequenceClassificationModel(model_type, model_name,
                                    use_cuda=False, num_labels=num_labels, args=model_args)


# Freeze BERT encoder
if freeze_bert_encoder:
    if model_type == 'camembert':
        for param in model.model.roberta.parameters():
            param.requires_grad = False
    elif model_type == 'flaubert':
        for param in model.model.transformer.parameters():
            param.requires_grad = False
    else:
        raise Exception('Cant freeze Bert encoder of %s' % model_type)

print_model(model.model, model_type)


if train:
    model.train_model(train_df, eval_df=eval_df)


if evaluate:
    print("\n\n#### Evaluation of model's performance on eval_df")
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print(result)
    # print(result, model_outputs, wrong_predictions)
    if isinstance(num_labels, list):
        preds_list = [np.argmax(pred, axis=1) if num_label > 1 else pred
                      for pred, num_label in zip(model_outputs, num_labels)]
    else:
        preds_list = [np.argmax(model_outputs, axis=1) if num_labels > 1 else model_outputs]
    # print(preds)
    # print(eval_df['labels'])
    print(eval_df_unzip.columns)
    for label_col, pred in zip(label_categories, preds_list):
        labels = eval_df_unzip[label_col]
        print("\n# Analyzing %s" % label_col)

        print('Labels mean: %s // var: %s' % (round(labels.mean(), 4), round(labels.var(), 4)))
        print('Preds mean: %s // var: %s' % (round(pred.mean(), 4), round(pred.var(), 4)))

        print('Median absolute error: %s' % round(median_absolute_error(labels, pred), 4))
        print('Mean absolute error: %s' % round(mean_absolute_error(labels, pred), 4))
        print('Mean squared error: %s' % round(mean_squared_error(labels, pred), 4))

        # If we are classifying with more than 2 possible values =>
        # encoder_label_name = model_args['output_dir'] + 'labels_encoders/' + label_col + '_classes.npy'
        # target_names = np.load(encoder_label_name, allow_pickle=True)
        # print('labels value_count')
        # print(labels.value_counts())
        # print('preds value_count')
        # print(np.array(np.unique(pred, return_counts=True)).T)
        # # print(type(target_names))
        # # print(target_names)
        # print(classification_report(labels, pred,
        #                             labels=range(len(target_names)), target_names=target_names))
        # print('# Confusion matrix')
        # print(confusion_matrix(labels, pred, labels=range(len(target_names))))

