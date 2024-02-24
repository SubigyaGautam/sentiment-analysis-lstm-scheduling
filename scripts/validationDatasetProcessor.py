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
import ast
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import math



# File path and nameE:/MS_CS/MS_CS_Assignments/1st Sem/Deep Learning CSCE598/Project/attempt2/
model_file_path = 'E:/MS_CS/MS_CS_Assignments/1st Sem/Deep Learning CSCE598/Project/working/state_dict_2_500.pth'
file_path_vocab = 'E:/MS_CS/MS_CS_Assignments/1st Sem/Deep Learning CSCE598/Project/attempt2/logs/vocabFile_test.txt'

file_path = 'E:/MS_CS/MS_CS_Assignments/1st Sem/Deep Learning CSCE598/Project/attempt2/validationLogs/logfile_test_500_2_5_2.txt'
file_path_predict = 'E:/MS_CS/MS_CS_Assignments/1st Sem/Deep Learning CSCE598/Project/attempt2/validationLogs/logfile_predict2_2.txt'




text_content = ''
# Open the file in append mode
log_file = open(file_path, 'w+')
# Open the file in append mode
log_file_predict = open(file_path_predict, 'a' ,  encoding='utf-8')


file_vocab = open(file_path_vocab, 'r')

log_file_predict.write(f'\n{datetime.datetime.now()} :: ---------------------------------------------------- Logging started when >=5.2--------------------------------------------')

import os

# Log some initial content
log_file.write(f"\n{datetime.datetime.now()} :: Logging started:")
for dirname, _, filenames in os.walk(''):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Importing Dataset

file_csv = 'E:/MS_CS/MS_CS_Assignments/1st Sem/Deep Learning CSCE598/Project/dataset/train.csv'
df_validation = pd.read_csv(file_csv, encoding = "ISO-8859-1",  names=["overall", "summary","text"])

df_validation['text'] = df_validation['text'] + ' ' + df_validation['summary']
del df_validation['summary']

print(df_validation.overall.value_counts())
df_validation = df_validation.loc[df_validation['overall'] != 3]
print(df_validation.overall.value_counts())

# # 650,000 - 100,00

# # 1    130000
# # 4    130000
# # 2    130000
# # 3    130000
# # 5    130000

# number_of_rows_to_delete = 100000

# # # # Identify rows where Column1 is equal to 1
# rows_to_delete_with_1 = df_validation[df_validation['overall'] == 1]
# rows_to_delete_with_2 = df_validation[df_validation['overall'] == 2]
# rows_to_delete_with_4 = df_validation[df_validation['overall'] == 4]
# rows_to_delete_with_5 = df_validation[df_validation['overall'] == 5]
# rows_to_delete_with_3 = df_validation[df_validation['overall'] == 3]




# # # Randomly select 10 rows to delete
# rows_to_delete_with_1 = rows_to_delete_with_1.sample(n=number_of_rows_to_delete)
# rows_to_delete_with_2 = rows_to_delete_with_2.sample(n=number_of_rows_to_delete)
# rows_to_delete_with_4 = rows_to_delete_with_4.sample(n=number_of_rows_to_delete)
# rows_to_delete_with_5 = rows_to_delete_with_5.sample(n=number_of_rows_to_delete)
# rows_to_delete_with_3 = rows_to_delete_with_3.sample(n=number_of_rows_to_delete)

# # # Drop the selected rows from the original DataFrame
# df_validation = df_validation.drop(rows_to_delete_with_1.index)
# df_validation = df_validation.drop(rows_to_delete_with_2.index)
# df_validation = df_validation.drop(rows_to_delete_with_4.index)
# df_validation = df_validation.drop(rows_to_delete_with_5.index)
# df_validation = df_validation.drop(rows_to_delete_with_3.index)



# # # Display the DataFrame after deletion
# print(f"\nDataFrame after deleting {number_of_rows_to_delete * 5} rows")

print(df_validation.overall.value_counts())
# print(df_validation.value_counts())

def sentiment_rating(rating):
    # Replacing ratings of 1,2,3 with 0 (not good) and 4,5 with 1 (good)
    if(int(rating) == 1 or int(rating) == 2):
        return 0
    else: 
        return 1
    
df_validation.overall = df_validation.overall.apply(sentiment_rating) 
print(df_validation.overall.value_counts())

rows_to_delete_with_0= df_validation[df_validation['overall'] == 0]
rows_to_delete_with_1 = df_validation[df_validation['overall'] == 1]

rows_to_delete_with_0 = rows_to_delete_with_0.sample(n=404688)
rows_to_delete_with_1 = rows_to_delete_with_1.sample(n=411833)

# # Drop the selected rows from the original DataFrame
df_validation = df_validation.drop(rows_to_delete_with_0.index)
df_validation = df_validation.drop(rows_to_delete_with_1.index)
print(df_validation.overall.value_counts())


# # DataFrame after deleting 1000000 rows
# # 3    12054
# # 5    11041
# # 4    10792
# # 2     8932
# # 1     5756

# # Name: overall, dtype: int64
# # 0    26742
# # 1    21833
# # Name: overall, dtype: int64

text_content = ''

vocabs = file_vocab.read()
vocab = ast.literal_eval(vocabs)

# Log some initial content
log_file_predict.write(f"\n{datetime.datetime.now()} :: Logging started:")
print(f"\n{datetime.datetime.now()} :: Logging started:")


# Model
no_layers = 2
vocab_size = 1000 + 1 #extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256
hidden_layer_batch_size = 8

# If we have a GPU available, we'll set our device to GPU. 

# # compose the LSTM Network

torch.manual_seed(1)
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available and is being used")
    log_file_predict.write(f' GPU is available and is being used ')

else:
    device = torch.device("cpu")
    print("GPU not available, CPU is being used")
    log_file_predict.write(f'GPU not available, CPU is being used')

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
        
    def forward(self, x, hidden):
        batch_size = 50

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
    


log_file_predict.write(f"\n{datetime.datetime.now()} :: Loading the model from: {model_file_path}")
print(f"\n{datetime.datetime.now()} :: Loading the model from: {model_file_path}")


model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5)


# Load the state dict if model is saved like : torch.save(model.state_dict(), model_file_path) 
# model.load_state_dict(torch.load(model_file_path))

# Load the model if model is saved like : torch.save(model , model_file_path)
model = torch.load(model_file_path)

# Set the model to evaluation mode
model.eval()  

# Log some initial content
log_file_predict.write(f"\n{datetime.datetime.now()} :: Prediction started:")
print(f"\n{datetime.datetime.now()} :: Prediction started:")

log_file_predict.write(f'\n{datetime.datetime.now()} :: Processing... ')
print(f'\n{datetime.datetime.now()} :: Processing... ')


#Finding stop words
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
log_file_predict.write(f'\n{datetime.datetime.now()} :: Processing... ')
print(f'\n{datetime.datetime.now()} :: Processing... ')

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
    


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

print(f'\n{datetime.datetime.now()} :: Processing... ')


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# we have very less number of reviews with length > 500. So we will consider only those below it.
# x_train_pad = padding_(x_train,500)
# x_test_pad = padding_(x_test,500)

#  Inference
def predict_text(text):
        # text = lemmatize_words(text)
        print(f'\n{datetime.datetime.now()} :: Predicting... ')

        log_file_predict.write(f'\n{datetime.datetime.now()} :: Word Sequence to predict sentiment of ::{text}')
        print(f'\n{datetime.datetime.now()} :: Word Sequence to predict sentiment of ::{text}')

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
        print(output.detach().cpu().numpy())
        mean =  np.mean(output.detach().cpu().numpy())
        print(f'Mean :: {mean}')
        return round(mean,2)



def predict(data,originalPrediction,predicted,tensorMeanMatrix):
    print( data)
    textToPredict = data['text']
    if isinstance(textToPredict, str) or isinstance(textToPredict, float) and not(math.isnan(textToPredict)):
        
        pro_tensor = predict_text(textToPredict)
        tensorMeanMatrix.append(pro_tensor)
        
        pred = 1 if pro_tensor >= 0.52 else 0
        originalPrediction.append(data['overall'])
        predicted.append(pred)

def f1_score_cal(originalPrediction, predicted):
        
     # Calculate confusion matrix
    conf_matrix = confusion_matrix(originalPrediction, predicted)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate metrics
    accuracy = accuracy_score(originalPrediction, predicted)
    precision = precision_score(originalPrediction, predicted)
    recall = recall_score(originalPrediction, predicted)
    f1 = f1_score(originalPrediction, predicted)

    # Calculate TPR and FPR
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    # Print the results
    print(f'\n{datetime.datetime.now()} :: ------------------------------------------------------------------------------------------------')
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"True Positive (TP): {tp}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Negative (TN): {tn}")
    print(f"True Positive Rate (TPR): {tpr}")
    print(f"False Positive Rate (FPR): {fpr}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    # Plotting
    labels = ['Precision', 'Recall', 'F1-Score']
    values = [precision, recall, f1]
    plt.figure(figsize = (8,6)) 

    plt.bar(labels, values, color=['blue', 'green', 'orange'])
    plt.ylim([0, 1])  # Set the y-axis limit between 0 and 1
    plt.title('Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.show()
    plt.savefig('Performance_Metrics_2.png')

    plt.figure(figsize=(8, 6))

    # Create confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])



    # Labeling quadrants
    plt.text(0.5, 1.5, f'\n\nTP', ha='center', va='center', color='black', fontsize=12)
    plt.text(1.5, 1.5, f'\n\nFN', ha='center', va='center', color='black', fontsize=12)

    plt.text(0.5, 0.5, f'\n\nFP', ha='center', va='center', color='black', fontsize=12)
    plt.text(1.5, 0.5, f'\n\nTN', ha='center', va='center', color='black', fontsize=12)


    # Plot confusion matrix using seaborn
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig('Confusion_Matrix_2.png')


    plt.show()

    # Print the results
    log_file_predict.write(f'\n{datetime.datetime.now()} :: ------------------------------------------------------------------------------------------------')
    log_file_predict.write(f"Confusion Matrix:\n{conf_matrix}")
    log_file_predict.write(f"True Positive (TP): {tp}")
    log_file_predict.write(f"False Positive (FP): {fp}")
    log_file_predict.write(f"False Negative (FN): {fn}")
    log_file_predict.write(f"True Negative (TN): {tn}")
    log_file_predict.write(f"True Positive Rate (TPR): {tpr}")
    log_file_predict.write(f"False Positive Rate (FPR): {fpr}")
    log_file_predict.write(f"Accuracy: {accuracy}")
    log_file_predict.write(f"Precision: {precision}")
    log_file_predict.write(f"Recall: {recall}")
    log_file_predict.write(f"F1-Score: {f1}")
    log_file_predict.write(f'\n{datetime.datetime.now()} :: ----------------------------------------- Logging ENDED when >=5.2-------------------------------------------------------')


    
    

originalPrediction = []
predicted = []
tensorMeanMatrix = []

# Iterate through the DataFrame
for index, row in df_validation.iterrows():
    predict(row,originalPrediction,predicted,tensorMeanMatrix)

if(len(originalPrediction) == len(predicted)):
    log_file_predict.write(f'\n{datetime.datetime.now()} :: originalPrediction\n')
    log_file_predict.write(f'\n{originalPrediction}\n')
    log_file_predict.write(f'\n{datetime.datetime.now()} :: tensorMeanMatrix\n')
    log_file_predict.write(f'\n{tensorMeanMatrix}\n')
    log_file_predict.write(f'\n{datetime.datetime.now()} :: predicted\n')
    log_file_predict.write(f'\n{predicted}\n')
    f1_score_cal(originalPrediction,predicted)


log_file_predict.close()
