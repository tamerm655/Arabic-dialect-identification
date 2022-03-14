# -*- coding: utf-8 -*-
"""
Fetching Data

"""

from google.colab import drive 
import os
import pandas as pd
import json
import requests

#loading data form drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/AIM_Task/dialect_dataset.csv"
df = pd.read_csv(path)

#Converting Id columns to a list
ids = list(df['id'])

#Converting IDs from integer to string
str_ids = []
for i in ids:
    str_ids.append(str(i))

#Splitting IDs to chunks of size 1000
def split(lst, chunk_size):

  for i in range(0, len(lst), chunk_size):
    yield json.dumps(lst[i:i + chunk_size])

chunk_size = 1000

lst = list(split(str_ids, chunk_size))

#Make request to get Json data
json_lst = []
for i in range(len(lst)):
    response = requests.post('https://recruitment.aimtechnologies.co/ai-tasks',  data=lst[i])
    json_lst.append(response.json())

#Mapping Json data to Dictionary
finalMap = {}
for d in json_lst:
    finalMap.update(d)

#Saving tweets to csv file
text_df = pd.DataFrame(finalMap.items(), columns=['id', 'text'])

text_df.to_csv('/content/drive/MyDrive/AIM_Task/requested_data.csv', index=False)
