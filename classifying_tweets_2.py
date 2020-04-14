import pandas as pd

dataframe_1 = pd.read_csv('tweets_data_2/betsentiment-FR-tweets-sentiment-teams.csv', encoding='iso-8859-1')
dataframe_2 = pd.read_csv('tweets_data_2/betsentiment-FR-tweets-sentiment-worldcup.csv', encoding='iso-8859-1')

print(dataframe_1.shape)
print(dataframe_1.columns)
print(pd.DataFrame([dataframe_1.sentiment.value_counts(),
                    dataframe_1.sentiment.value_counts()/dataframe_1.shape[0]]))

print(dataframe_2.shape)
print(dataframe_2.columns)
print(pd.DataFrame([dataframe_2.sentiment.value_counts(),
                    dataframe_2.sentiment.value_counts()/dataframe_2.shape[0]]))