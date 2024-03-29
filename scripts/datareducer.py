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



text_content = ''

import os

# Importing Dataset

file_csv = '../dataset/Big_dataset.csv'
# df_train = pd.read_csv(file_csv, encoding = "ISO-8859-1",  names=["overall", "summary", "reviewText"])
df_train = pd.read_csv(file_csv, encoding = "ISO-8859-1",  names=["rating","overall", "text"])



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



# # Randomly select  rows to delete
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

print(df_train.rating.value_counts())
print(df_train.value_counts())

def sentiment_rating(rating):
    # Replacing ratings of 1,2 with 0 (not good) and 4,5 with 1 (good)
    if int(rating) == 1 or int(rating) == 2:
        return 0
    elif int(rating) == 3:
        return None  # Return None for ratings of 3 (to be removed later)
    else:
        return 1

# Apply the sentiment_rating function to transform the ratings column
df_train['rating'] = df_train['rating'].apply(sentiment_rating)

# Remove rows with a rating of 3
df_train = df_train[df_train['rating'].notna()]

# Shuffle the DataFrame
df_train = df_train.sample(frac=1, random_state=42)

# Identify and delete 1000 rows with a rating value of 1
rows_to_delete_1 = df_train[df_train['rating'] == 0].index[:1000]
rows_to_delete_2 = df_train[df_train['rating'] == 1].index[:1000]

df_train = df_train.drop(rows_to_delete_1)
df_train = df_train.drop(rows_to_delete_2)

print(df_train.rating.value_counts())
print(df_train.value_counts())

# Write the processed DataFrame to a new CSV file
df_train.to_csv('processed_data.csv', index=False)