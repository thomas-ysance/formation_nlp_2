import pandas as pd

xaa = pd.read_csv('../French-Sentiment-Analysis-Dataset/xaa', sep='ùù')


print(xaa.shape)
print(xaa.head())
# xaa
# xaa['text'] = xaa['polarity,statutnull'].str.split(',', 2)[1]
# xaa['text'] = xaa['polarity,statutnull'].apply(lambda x: x.split(',', 1)[1])


def splitting(text):

    print(text)
    only_text = text.split(",", 1)[1]
    print(only_text)
    return only_text


xaa['text'] = xaa['polarity,statutnull'].apply(splitting)


print(xaa.shape)
print(xaa.head())
