import numpy as np # linear algebra
import pandas as pd
import json

import seaborn as sns
from sklearn.feature_selection import SequentialFeatureSelector # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize,sent_tokenize
import string
import re
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import time
import datetime;


# File path and name
file_path = 'logs/logfile_test_500.txt'
training_file_path = 'logs/trainingData_test_500.txt'
testing_file_path = 'logs/testingData_test_500.txt'
tensor_data_file_path_train = 'logs/tesorDataTest_test_500.txt'
tensor_data_file_path_test = 'logs/tesorDataTrain_test_500.txt'
training_log = 'logs/trainingLog_test_500.txt'
vocab_file_path = 'logs/vocabFile_test_500.txt'


text_content = ''
# Open the file in append mode
log_file = open(file_path, 'w+')
log_file_training = open(training_file_path, 'w+')
log_file_testing = open(testing_file_path, 'w+')
log_file_tensor_train = open(tensor_data_file_path_train, 'w+')
log_file_tensor_test = open(tensor_data_file_path_test, 'w+')
log_file_training_log = open(training_log, 'w+')
vocab_file = open(vocab_file_path, 'w+')


log_file.truncate(0)
log_file_training.truncate(0)
log_file_testing.truncate(0)
log_file_tensor_train.truncate(0)
log_file_tensor_test.truncate(0)
log_file_training_log.truncate(0)

import os

# Log some initial content
log_file.write(f"\n{datetime.datetime.now()} :: Logging started:")
for dirname, _, filenames in os.walk(''):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Importing Dataset

file_csv = 'output_file_reduced_data.csv'
# df_train = pd.read_csv(file_csv, encoding = "ISO-8859-1",  names=["overall", "summary", "reviewText"])
df_train = pd.read_csv(file_csv, encoding = "ISO-8859-1",  names=["overall", "text"])



# print(df_train.head())

# # Removing the unwanted columns from the dataframe and combining reviewText and summary column to one column and naming it as text

# del df_train['reviewText']
# del df_train['summary']

print(df_train.overall.value_counts())
# 650,000 - 100,00

# 1    130000
# 4    130000
# 2    130000
# 3    130000
# 5    130000

# number_of_rows_to_delete = 120000
# # Display the original DataFrame
print("Original DataFrame:")
print(df_train)


# # Identify rows where Column1 is equal to 1
# rows_to_delete_with_1 = df_train[df_train['overall'] == 1]
# rows_to_delete_with_2 = df_train[df_train['overall'] == 2]
# rows_to_delete_with_4 = df_train[df_train['overall'] == 4]
# rows_to_delete_with_5 = df_train[df_train['overall'] == 5]



# # Randomly select 10 rows to delete
# rows_to_delete_with_1 = rows_to_delete_with_1.sample(n=number_of_rows_to_delete)
# rows_to_delete_with_2 = rows_to_delete_with_2.sample(n=number_of_rows_to_delete)
# rows_to_delete_with_4 = rows_to_delete_with_4.sample(n=number_of_rows_to_delete)
# rows_to_delete_with_5 = rows_to_delete_with_5.sample(n=number_of_rows_to_delete)


# # Drop the selected rows from the original DataFrame
# df_train = df_train.drop(rows_to_delete_with_1.index)
# df_train = df_train.drop(rows_to_delete_with_2.index)
# df_train = df_train.drop(rows_to_delete_with_4.index)
# df_train = df_train.drop(rows_to_delete_with_5.index)


# # Display the DataFrame after deletion
# print(f"\nDataFrame after deleting {number_of_rows_to_delete * 4} rows")

print(df_train.overall.value_counts())
print(df_train.value_counts())


def sentiment_rating(rating):
    # Replacing ratings of 1,2,3 with 0 (not good) and 4,5 with 1 (good)
    if(int(rating) == 1 or int(rating) == 2 or int(rating) == 3):
        return 0
    else: 
        return 1
    
# df_train.overall = df_train.overall.apply(sentiment_rating) 



data = df_train  #  dataset
labels = ['overall','text']

# Save the DataFrame to a CSV file
# df_train.to_csv('output_file_reduced_data.csv', index=False)

class BuildDataSet(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Implement logic to return a sample
        # Return a tuple (sample, target) or a dictionary {'data': sample, 'target': target}
        return (self.data['overall'][idx], self.data['text'][idx])


#Finding stop words
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Lemmatizing the words
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    final_text = []
    if isinstance(text, str):  # Check if text is a string
        for i in text.split():
            
            if i.strip().lower() not in stop:
                pos = pos_tag([i.strip()])
                word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
                final_text.append(word.lower())
        return " ".join(final_text)
    else:
     return " " 
    
    
df_train.text = df_train.text.apply(lemmatize_words)
print('-------------- After lemmatization ------------------')
print(df_train.head)

# log_file.write(f'\n{datetime.datetime.now()} :: -------------- After lemmatization ------------------')
# log_file.write(f" {datetime.datetime.now()} :: {df_train.head }")

# # -------------- Affter lemmatization ------------------
# # <bound method Ndf_trainrame.head of        overall                                               text
# # 0            1  much write here, exactly suppose to. filter po...
# # 1            1  product exactly quite affordable.i realize dou...
# # 2            1  primary job device block breath would otherwis...
# # 3            1  nice windscreen protects mxl mic prevents pops...
# # 4            1  pop filter great. look performs like studio fi...
# # ...        ...                                                ...
# # 10256        1             great, expected. thank all. five stars
# # 10257        1  i've think try nanoweb string while, bit put h...
# # 10258        1  try coat string past include elixirs) never fo...
# # 10259        1  well, made elixir developed taylor guitars ......
# # 10260        1  string really quite good, call perfect. unwoun...

# # [10261 rows x 2 columns]>

# Split the dataset into training and testing indices
# x_train, x_test, y_train, y_test = train_test_split(list(range(len(dataset_to_use))), test_size=0.2, random_state=42)
x_train,x_test,y_train,y_test = train_test_split(df_train.text, df_train.overall,test_size = 0.2 , random_state = 0)

def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tockenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(stopwords.words('english')) 
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
  
    corpus = Counter(word_list)

    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]

    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
    
    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                    if preprocess_string(word) in onehot_dict.keys()])
            
    encoded_train = [1 if label == 1 else 0 for label in y_train]  
    encoded_test = [1 if label == 0 else 0 for label in y_val] 
    #
    #
    #
    # Assuming final_list_train, final_list_test are lists of sequences of varying lengths
    # Pad sequences to make them uniform in length
    # max_length = max(len(seq) for seq in final_list_train + final_list_test)
    # final_list_train = [seq + [0] * (max_length - len(seq)) for seq in final_list_train]
    # final_list_test = [seq + [0] * (max_length - len(seq)) for seq in final_list_test]

    print('\n-------------final_list_train-------------------')
    print(final_list_train)

    # Convert each inner list to a string
    list_strings_final_list_train = [str(sublist) for sublist in final_list_train]
    # Merge the list representations into a single string
    string_final_list_train = '\n'.join(list_strings_final_list_train)


    list_strings_final_list_test = [str(sublist) for sublist in final_list_test]
    string_final_list_test = '\n'.join(list_strings_final_list_test)


    list_strings_encoded_test = [str(sublist) for sublist in encoded_test]
    string_final_encoded_test = ' , '.join(list_strings_encoded_test)


    list_strings_onehot_dict = [str(sublist) for sublist in onehot_dict]
    string_final_onehot_dict = ' , '.join(list_strings_onehot_dict)
    log_file.write(string_final_onehot_dict)

    # Convert to NumPy arrays
    np_final_list_train = np.array(final_list_train)
    np_final_list_test = np.array(final_list_test)

    return np_final_list_train, np.array(encoded_train), np_final_list_test, np.array(encoded_test), onehot_dict

good = x_train[y_train[y_train == 1].index]
bad = x_train[y_train[y_train == 0].index]

x_train.shape, good.shape, bad.shape
print('-------------shapes-------------------')
print(x_train.shape, y_train.shape, good.shape, bad.shape)
# ((8208,), (7197,), (1011,))

print('-------------- train_indices ------------------')
print(len(y_train)) 
# 8208

print('-------------- test_indices ------------------')
print(len(y_test)) 
# 2053


log_file.write(f'\n{datetime.datetime.now()} :: -------------shapes-------------------')
log_file.write(f" {datetime.datetime.now()} ::{x_train.shape, y_train.shape, good.shape, bad.shape}")
# ((8208,), (7197,), (1011,))
log_file.write('-------------good-------------------')
log_file.write(f" {datetime.datetime.now()} ::{good}")
log_file.write('-------------bad-------------------')
log_file.write(f" {datetime.datetime.now()} ::{bad}")

log_file.write(f'\n{datetime.datetime.now()} :: -------------- train_indices ------------------')
log_file.write(f" {datetime.datetime.now()} ::{len(y_train)}") 
# 8208

log_file.write(f'\n{datetime.datetime.now()} :: -------------- test_indices ------------------')
log_file.write(f" {datetime.datetime.now()} ::{len(y_test)}") 

# 2053


# Text Reviews with Poor Ratings
plt.figure(figsize = (20,20)) 
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(bad))
plt.imshow(wc,interpolation = 'bilinear')
plt.title('Text Reviews with Bad Ratings')
plt.savefig('Text_Bad_Ratings_test_500.png')

# Text Reviews with Good Ratings
plt.figure(figsize = (20,20)) 
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(good))
plt.imshow(wc,interpolation = 'bilinear')
plt.title('Text Reviews with Good Ratings')
plt.savefig('Text_Good_Ratings_test_500.png')

plt.figure(figsize = (20,20)) 
dd = pd.Series(y_train).value_counts()
print(f'\n{datetime.datetime.now()} :: dd is : {dd}')

sns.barplot(x=np.array(['positive','negative']),y=dd.values)
plt.savefig('Good Ratings Vs Bad Ratings_test_500.png')




# Length of vocabulary is 1000

# Analysing review length
rev_len = [len(i) for i in x_train]
print(f'rev_len is : {rev_len}')
plt.figure(figsize = (20,20)) 
pd.Series(rev_len).hist()
plt.savefig('ReviewLength.png')
print(f'Description of the data : {pd.Series(rev_len).describe()}')
log_file.write(f'\n{datetime.datetime.now()} :: Description of the data :\n {pd.Series(rev_len).describe()} ')
# Description of the data : 
# count    8208.000000
# mean       36.255970
# std        40.175509
# min         0.000000
# 25%        15.000000
# 50%        24.000000
# 75%        42.000000
# max       745.000000
# dtype: float64

# count    32000.000000
# mean       284.041812
# std        158.553275
# min          1.000000
# 25%        151.000000
# 50%        250.000000
# 75%        392.000000
# max       1018.000000

# Tokenize after representation
x_train, y_train, x_test, y_test, vocab = tockenize(x_train, y_train, x_test, y_test)

print(f'Length of vocabulary is {len(vocab)}')
log_file.write(f'\n {datetime.datetime.now()} :: Length of vocabulary is : {len(vocab)} ')
vocab_file.write(json.dumps(vocab))




# Observations :
# a) Mean review length = around 36.
# b) minimum length of reviews is 0.
# c)There are quite a few reviews that are extremely long, we can manually investigate them to check whether we need to include or exclude them from our analysis.

# padding each of the sequence to max length
def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# count    32000.000000
# mean       284.041812
# std        158.553275
# min          1.000000
# 25%        151.000000
# 50%        250.000000
# 75%        392.000000
# max       1018.000000
x_train_pad = padding_(x_train,500)
x_test_pad = padding_(x_test,500)

# creating Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

# dataloaders
#  changing batch size to 48 because our data 8208 is perfectly divisible by 48 , 10267 / 31  = 331
batch_size = 50

# Assuming train_data is your original dataset
original_dataset_size = len(train_data)
original_dataset_size_val = len(valid_data)



# # Number of data points to remove
# num_data_points_to_remove = 8
# num_data_points_to_remove_val = 3

# # Indices of data points to remove
# indices_to_remove = torch.randperm(original_dataset_size)[:num_data_points_to_remove]
# indices_to_remove_v = torch.randperm(original_dataset_size_val)[:num_data_points_to_remove_val]


# # Remove the selected indices from the dataset
# filtered_dataset = [data_point for i, data_point in enumerate(train_data) if i not in indices_to_remove]
# filtered_datase_v = [data_point for i, data_point in enumerate(valid_data) if i not in indices_to_remove_v]


# # Create a new DataLoader with the filtered dataset
# # making sure to SHUFFLE the data

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

for data, label in train_data:
    data_str = ','.join([str(item.item()) for item in data])
    log_file_tensor_train.write(f" {datetime.datetime.now()} ::{data_str}, {label.item()}")

for data, label in valid_data:
    data_str = ','.join([str(item.item()) for item in data])
    log_file_tensor_test.write(f" {datetime.datetime.now()} ::{data_str}, {label.item()}")

# obtaining one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)

# obtaining one batch of training data
dataiter_v = iter(valid_loader)
sample_x_v, sample_y_v = next(dataiter_v)

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print('Sample input: \n', sample_y)

print('Sample input size: (valid) ', sample_x_v.size()) # batch_size, seq_length
print('Sample input (valid): \n', sample_x_v)
print('Sample input (valid): \n', sample_y_v)

log_file.write(f' \n{datetime.datetime.now()} ::  Sample input size: : {sample_x.size()} ')
log_file.write(f' \n{datetime.datetime.now()} ::  Sample input size: sample_x : {sample_x} ')
log_file.write(f' \n{datetime.datetime.now()} ::  Sample input size: sample_y : {sample_y} ')

# Sample input size:  torch.Size([50, 500])
# Sample input: 
#  tensor([[  0,   0,   0,  ..., 106, 600, 183],
#         [  0,   0,   0,  ...,  90, 661,   7],
#         [  0,   0,   0,  ..., 172,  79, 367],
#         ...,
#         [  0,   0,   0,  ...,  44,   3,  38],
#         [  0,   0,   0,  ...,  48,  14, 262],
#         [  0,   0,   0,  ..., 894,   3,  28]], dtype=torch.int32)
# Sample input:
#  tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
#         0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
#         1, 1], dtype=torch.int32)

# Model
no_layers = 2
vocab_size = len(vocab) + 1 #extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256
hidden_layer_batch_size = 8
# We need to add an embedding layer because there are less words in our vocabulary.
# It is massively inefficient to one-hot encode that many classes. So, instead of one-hot encoding,
# we can have an embedding layer and use that layer as a lookup table. 
# You could train an embedding layer using Word2Vec,
# then load it here. But, it's fine to just make a new layer, using it for only dimensionality reduction, 
# and let the network learn the weights.

class SentimentRNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5):
        super(SentimentRNN,self).__init__()
 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        self.no_layers = no_layers
        self.vocab_size = vocab_size
    
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm
        self.lstm = nn.LSTM( input_size = embedding_dim,
                             hidden_size = self.hidden_dim,
                             num_layers = no_layers,
                             batch_first = True)
        
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
    
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        log_file_training_log.write(f'\n{datetime.datetime.now()} -------------------------------__init__-------------------------------------------------')

        
    def forward(self, x, hidden):
        batch_size = 50

        print(f' forward : batch_size  , x.size(0) : {batch_size} , {x.size(0)}')
        log_file_training_log.write(f'\n{datetime.datetime.now()} ::  forward : batch_size  , x.size(0) : {batch_size} , {x.size(0)}')

        # embeddings and lstm_out
        embeds = self.embedding(x)
        
        # shape: B x S x Feature   since batch = True
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

# If we have a GPU available, we'll set our device to GPU. 

# # compose the LSTM Network

torch.manual_seed(1)
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
    log_file.write(f' GPU is available ')

else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    log_file.write(f'GPU not available, CPU used ')


# Building model
log_file.write(f'\n{datetime.datetime.now()} ::  Building Model:')
log_file.write(f'\n{datetime.datetime.now()} ::  no_layers : {no_layers}')
log_file.write(f'\n{datetime.datetime.now()} ::  vocab_size : {vocab_size}')
log_file.write(f'\n{datetime.datetime.now()} ::  hidden_dim : {hidden_dim}')
log_file.write(f'\n{datetime.datetime.now()} ::  embedding_dim : {embedding_dim}')
log_file.write(f'\n{datetime.datetime.now()} :: drop_prob : 0.5 ')

print(f' Building Model:')
print(f' no_layers : {no_layers}')
print(f' vocab_size : {vocab_size}')
print(f' hidden_dim : {hidden_dim}')
print(f' embedding_dim : {embedding_dim}')
print(f' drop_prob : 0.5 ')
model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5)

#moving to gpu
model.to(device)

print(model)
log_file.write(f' \n model: : {model} ')


print(f"\n{datetime.datetime.now()} :: Logging completed. File '{file_path}' has been appended with new content.")
################################################################

#  Training

# Log some initial content
log_file_training_log.write(f"\n{datetime.datetime.now()} :: ------------------------------- Training started ------------------------------- ")
print("\n ------------------------------- Training started ------------------------------- ")

lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

clip = 5
epochs = 5 
valid_loss_min = np.Inf

# train for some number of epochs
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]

print("\n{datetime.datetime.now()} :: ------------------------------- Entering epoch ------------------------------- ")
log_file_training_log.write(f'\n{datetime.datetime.now()} :: ------------------------------- Entering epoch : {epochs} ------------------------------- ')

for epoch in range(epochs):
    print(f'Epoch {epoch+1}') 
    log_file_training_log.write(f"\n{datetime.datetime.now()} :: ------------------------------- Training started ------------------------------- ")
    log_file_training_log.write(f'\n{datetime.datetime.now()} :::: Epoch :  {epoch+1}') 


    train_losses = []
    train_acc = 0.0
    model.train()

    # initialize hidden state 
    batch_size = 50
    h = model.init_hidden(batch_size)

    count = 0
    for inputs, labels in train_loader:
        count = count + 1
        log_file_training_log.write(f'\n{datetime.datetime.now()}::------------------for inputs, labels in train_loader:) : iteration {count}-------------------------------------------- ')

        inputs, labels = inputs.to(device), labels.to(device)   
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        
        model.zero_grad()

        output,h = model(inputs,h)
        
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        train_losses.append(loss.item())

        # calculating accuracy
        accuracy = acc(output,labels)
        train_acc += accuracy

        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        log_file_training_log.write(f'\n{datetime.datetime.now()} :: last step of (for inputs, labels in train_loader: ------------------------------------ ')

    log_file_training_log.write(f'\n{datetime.datetime.now()} :: Exit (for inputs, labels in train_loader):-------------------------------------------- ')
    print(f'\n{datetime.datetime.now()} :: Exit (for inputs, labels in train_loader):-------------------------------------------- ')

    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()
    countL = 0
    for inputs, labels in valid_loader:
            countL = countL +1

            val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())
            
            accuracy = acc(output,labels)
            val_acc += accuracy

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc/len(train_loader.dataset)
    epoch_val_acc = val_acc/len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)

    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')

    log_file_training_log.write(f'\n{datetime.datetime.now()} :: Epoch {epoch+1}') 
    log_file_training_log.write(f'\n{datetime.datetime.now()} :: train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    log_file_training_log.write(f'\n{datetime.datetime.now()} :: train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')

    if epoch_val_loss <= valid_loss_min:
        # Create the directory if it does not exist
        # directory = '/working'
        # os.makedirs(directory, exist_ok=True)

        # # Your model saving code
        # torch.save(model.state_dict(), os.path.join(directory, 'state_dict.pth'))
        # relative_path = '1st Sem/Deep Learning CSCE598/Project/working/state_dict.pth'
        file_path_1 = 'D:/Subigya/MS_CS_Assignments/1st Sem/Deep Learning CSCE598/Project/working/state_dict_1_500.pth'
        file_path_2 = 'D:/Subigya/MS_CS_Assignments/1st Sem/Deep Learning CSCE598/Project/working/state_dict_2_500.pth'

        # when you want to save only the parameters and need to rebuild the model separately.
        torch.save(model.state_dict(), file_path_1) 

        # when you want to save the entire model, including its architecture, 
        # and plan to load it for further training or inference in the same environment where the model class is defined.
        torch.save(model , file_path_2)

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
        log_file_training_log.write('\n Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
        print(f'\n Model is saved : {os.getcwd()}')

        valid_loss_min = epoch_val_loss
    print(25*'==')
    log_file_training_log.write(25*'==')


fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 2, 1)
plt.plot(epoch_tr_acc, label='Train Acc')
plt.plot(epoch_vl_acc, label='Validation Acc')
plt.legend()
plt.grid()
plt.title("Accuracy")
plt.savefig('Accuracy_500.png')

    
plt.subplot(1, 2, 2)
plt.plot(epoch_tr_loss, label='Train loss')
plt.plot(epoch_vl_loss, label='Validation loss')
plt.title("Loss")
plt.legend()
plt.grid()
plt.savefig('Loss_500.png')


#  Inference
def predict_text(text):
        word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
                        if preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq, axis=0)
        pad = torch.from_numpy(padding_(word_seq, 500))
        inputs = pad.to(device)
        batch_size = 1
        h = model.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
        
        # Instead of returning a Python scalar, return the entire tensor or convert it to a NumPy array 
        # return(output.item())
        print(f'\n {datetime.datetime.now()} output :: {output}')
        return output.detach().cpu().numpy()[0]


# review	sentiment
# One of	positive

# overall    text
# 1  much write here, exactly suppose to. filter po

index = 30
print(df_train['text'][index])
print('=' * 70)
print(f'Actual sentiment is: {df_train["overall"][index]}')
print('=' * 70)
pro_tensor = predict_text(df_train['text'][index])  # Notice the variable name change to avoid conflict with 'pro'
status = "positive" if pro_tensor > 0.5 else "negative"
pro_tensor = (1 - pro_tensor) if status == "negative" else pro_tensor

pro = pro_tensor.item()  # Convert to Python scalar if needed
print(f'Predicted sentiment is {status} with a probability of {pro}')
log_file_training_log.write(f'\n{datetime.datetime.now()} :: Predicted sentiment is {status} with a probability of {pro}')

index = 32
print(df_train['text'][index])
print('='*70)
print(f'Actual sentiment is  : {df_train["overall"][index]}')
print('='*70)
pro_tensor_2 = predict_text(df_train['text'][index])
status = "positive" if pro_tensor_2 > 0.5 else "negative"
pro_tensor_2 = (1 - pro_tensor_2) if status == "negative" else pro_tensor_2

pro = pro_tensor.item()  # Convert to Python scalar if needed
print(f'predicted sentiment is {status} with a probability of {pro}')
log_file_training_log.write(f'\n{datetime.datetime.now()} :: Predicted sentiment is {status} with a probability of {pro}')


log_file.close()
log_file_training_log.close()