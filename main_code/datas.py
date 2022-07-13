import numpy as np

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
import os
import ast
from sklearn.utils import shuffle
import random


def make_dataset(csv_file, tokenizer, max_length=512, random_state=1000, data_cut=None):
        ''' data load '''
        ''' 1기+2기 데이터 '''
        #data = csv_file
        #total_data = pd.read_csv(data)

        ''' 재선이형이 준 데이터 '''
        total_data = pd.read_csv(csv_file)
        total_data.columns=['paragraph', 'category']
        label_dict = {'Abstract':0, 'Introduction':1, 'Main':2, 'Methods':3, 'Summary':4, 'Captions':5}
        total_data['label'] = total_data.category.replace(label_dict)
        
        if not data_cut is None:
            total_data = total_data.iloc[:data_cut,:]
            
        total_text = total_data['paragraph'].to_list()
        total_label = total_data['label'].to_list()

        ''' type error 방지 '''
        if type(total_label[0]) == str:
            total_label = [ast.literal_eval(l) for l in total_label]

        if type(total_label[0]) == int:
            total_label = np.eye(6)[total_label].tolist()

        train_text, val_text, train_labels, val_labels = train_test_split(total_text, total_label, test_size=0.2, random_state=random_state, stratify=total_label)

        ''' data들 tokenizing '''
        train_encodings= tokenizer.batch_encode_plus(train_text, truncation=True, return_token_type_ids=True, max_length=max_length, add_special_tokens=True, return_attention_mask=True, padding='max_length')
        val_encodings = tokenizer.batch_encode_plus(val_text, truncation=True, return_token_type_ids=True, max_length=max_length, add_special_tokens=True, return_attention_mask=True, padding='max_length')

        ''' token tensor 화 '''
        train_encodings = {key: torch.tensor(val) for key, val in train_encodings.items()}
        val_encodings = {key: torch.tensor(val) for key, val in val_encodings.items()}

        ''' labels tensor 화 '''
        train_labels_ = {}
        train_labels_['label_onehot'] = torch.tensor(train_labels, dtype=torch.float)
        train_labels_['label'] = torch.tensor([t.index(1) for t in train_labels], dtype=torch.int)
        train_labels = train_labels_

        val_labels_ = {}
        val_labels_['label_onehot'] = torch.tensor(val_labels, dtype=torch.float)
        val_labels_['label'] = torch.tensor([t.index(1) for t in val_labels], dtype=torch.long)
        val_labels = val_labels_

        ''' dataset class 생성 '''
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels, texts):
                self.encodings = encodings
                self.labels = labels
                self.texts = texts

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['text'] = self.texts[idx]
                # scalar version
                item['label'] = self.labels['label'][idx]
                # one-hot version
                item['label_onehot'] = self.labels['label_onehot'][idx]
                return item

            def __len__(self):
                return len(self.labels['label_onehot'])

        ''' train을 위한 format으로 data들 변환 '''
        train_dataset = CustomDataset(train_encodings, train_labels, train_text)
        val_dataset = CustomDataset(val_encodings, val_labels, val_text)
        
        return train_dataset, val_dataset