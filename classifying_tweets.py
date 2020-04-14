import pandas as pd
from os import listdir
from os.path import isfile, join


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
print(all_tweets.shape)
print(all_tweets.describe())
