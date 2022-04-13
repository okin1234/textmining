from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch
import os
import ast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import EarlyStoppingCallback
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils import shuffle
from transformers import get_cosine_schedule_with_warmup
from torch.nn import functional as F
import random
from .datas import make_dataset
from .utils import set_seed, accuracy_per_class, compute_metrics, model_eval, checkpoint_save, EarlyStopping, model_freeze
from .model import classification_model


    
class NLP_classification():
    def __init__(self, model_name=None, max_length=512, random_state=1000, task_type='onehot', freeze_layers=None, num_classifier=1):
        self.model_name = model_name
        self.max_length = max_length
        self.random_state = random_state
        self.task_type = task_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.config = AutoConfig.from_pretrained(model_name, num_labels=6)
        #self.pretrained_model = AutoModelForSequenceClassification.from_config(self.config)
        self.pretrained_model = AutoModel.from_config(self.config)
        self.freeze_layers=freeze_layers
        self.num_classifier=num_classifier
        
        
    def training(self, data_file, epochs=50, batch_size=4, lr=1e-5, dropout=0.1, data_cut=None,
                wandb_log=False, wandb_project=None, wandb_group=None, wandb_name=None, wandb_memo=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(self.random_state)
        torch.set_num_threads(10)
        task_type=self.task_type
        
        if wandb_log is True:
            import wandb
            wandb.init(project=wandb_project, reinit=True, group=wandb_group, notes=wandb_memo)
            wandb.run.name = wandb_name
            wandb.run.save()
            parameters = wandb.config
            parameters.lr = lr
            parameters.batch_size = batch_size
            parameters.dropout = dropout
            parameters.train_num = data_cut
            parameters.max_length = self.max_length
            parameters.model_name = self.model_name
            parameters.task_type = self.task_type
        
        '''data loading'''
        train_dataset, val_dataset = make_dataset(csv_file=data_file, tokenizer=self.tokenizer, max_length=self.max_length, random_state=self.random_state, data_cut=data_cut)
        
        '''loader making'''
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset))

        ''' model load '''
        model=classification_model(self.pretrained_model, self.config, num_classifier=self.num_classifier)
        model=model_freeze(model, self.freeze_layers)
        model.to(device)
        
        ''' running setting '''
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, eps=1e-8)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=(len(train_loader)*epochs))
        early_stopping = EarlyStopping(patience = 10, verbose = True)
        
        ''' running '''
        best_epoch = None
        best_val_loss = None

        for epoch in range(epochs):
            model.train()
            loss_all = 0
            step = 0

            for data in tqdm(train_loader):
                input_ids=data['input_ids'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                if task_type=='onehot':
                    targets=data['label_onehot'].to(device, dtype=torch.float)
                elif task_type=='scalar':
                    targets=data['label'].to(device, dtype=torch.long)

                inputs = {'input_ids': input_ids, 'attention_mask': mask,
                  'labels': targets}

                outputs = model(inputs)
                output = outputs[1]
                loss = outputs[0]

                optimizer.zero_grad()
                #loss=loss_fn(output, targets)
                loss_all += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                #print(optimizer.param_groups[0]['lr'])

            train_loss = loss_all/len(train_loader)
            val_loss, val_acc, val_precision, val_recall, val_f1 = model_eval(model, device, val_loader, task_type=task_type)
            
            if wandb_log is True:
                wandb.log({'train_loss':train_loss, 'val_loss':val_loss, 'val_acc':val_acc,
                           'val_precision':val_precision, 'val_recall':val_recall, 'val_f1':val_f1})

            if best_val_loss is None or val_loss <= best_val_loss:
                best_epoch = epoch+1
                best_val_loss = val_loss
                checkpoint_save(model, val_loss, wandb_name=wandb_name)
                
            print('Epoch: {:03d}, Train Loss: {:.7f}, Val Loss: {:.7f}, Val Acc: {:.7f}, Val Precision: {:.7f}, Val Recall: {:.7f}, Val F1: {:.7f} '.format(epoch+1, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1))
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break


        

    