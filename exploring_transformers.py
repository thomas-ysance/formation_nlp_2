import torch
from transformers import FlaubertModel, FlaubertTokenizer

# Choose among ['flaubert-small-cased', 'flaubert-base-uncased', 'flaubert-base-cased', 'flaubert-large-cased']
modelname = 'flaubert-small-cased'

# Load pretrained model and tokenizer
flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
# do_lowercase=False if using cased models, True if using uncased ones

sentence = "Le chat mange une pomme."
token_ids = torch.tensor([flaubert_tokenizer.encode(sentence)])
print(token_ids.shape)

last_layer = flaubert(token_ids)[0]
print(last_layer.shape)
# torch.Size([1, 8, 768])  -> (batch size x number of tokens x embedding dimension)

# The BERT [CLS] token correspond to the first hidden state of the last layer
cls_embedding = last_layer[:, 0, :]
print(cls_embedding.shape)
