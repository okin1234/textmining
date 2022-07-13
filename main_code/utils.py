from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import shuffle
import random
import datetime as dt
import os
from glob import glob

def checkpoint_save(model, val_loss, checkpoint_dir=None, wandb_name=None):
    if checkpoint_dir is None:
        checkpoint_dir = './save_model'
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    x = dt.datetime.now()
    y = x.year
    m = x.month
    d = x.day
    
    if wandb_name is None:
        wandb_name = "testing"
    
    torch.save(model.state_dict(), "./save_model/{}_{}_{}_{:.4f}_{}.pt".format(y, m, d, val_loss, wandb_name))
    
    #saved_dict_list = glob(os.path.join(checkpoint_dir, '*.pt'))
    saved_dict_list = glob(os.path.join(checkpoint_dir, '{}_{}_{}_*_{}.pt'.format(y,m,d,wandb_name)))
    
    
    val_loss_list = np.array([float(os.path.basename(loss).split("_")[3]) for loss in saved_dict_list])
    saved_dict_list.pop(val_loss_list.argmin())
    
    for i in saved_dict_list:
        os.remove(i)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def accuracy_per_class(preds, labels):
    label_dict = {'Abstract':0, 'Intro':1, 'Main':2, 'Method':3, 'Summary':4, 'Caption':5}
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    class_list = []
    acc_list = []
    for label in list(label_dict.values()):
        y_preds = preds[labels==label]
        y_true = labels[labels==label]
        class_list.append(label_dict_inverse[label])
        acc_list.append("{0}/{1}".format(len(y_preds[y_preds==label]), len(y_true)))
    
    print("{:10} {:10} {:10} {:10} {:10} {:10}".format(class_list[0], class_list[1], class_list[2], class_list[3], class_list[4], class_list[5]))
    print("{:10} {:10} {:10} {:10} {:10} {:10}".format(acc_list[0], acc_list[1], acc_list[2], acc_list[3], acc_list[4], acc_list[5]))    
    

def compute_metrics(output, target, task_type='onehot'):
    if task_type=='onehot':
        pred=np.argmax(output, axis=1).flatten()
        labels=np.argmax(target, axis=1).flatten()
    elif task_type=='scalar':
        pred=np.argmax(output, axis=1).flatten()
        labels=np.array(target).flatten()
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro', zero_division=0)
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    
    accuracy_per_class(pred, labels)
        
    return [accuracy, precision, recall, f1]


def model_eval(model, device, loader, task_type='onehot', return_values=False):
    model.eval()
    error = 0
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    eval_targets=[]
    eval_outputs=[]
    eval_texts=[]
    with torch.no_grad():
        for data in tqdm(loader):
            eval_texts.extend(data['text'])
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
            #loss=loss_fn(output, targets)
            error+=loss
            #output = torch.sigmoid(output)
            eval_targets.extend(targets.detach().cpu().numpy())
            eval_outputs.extend(output.detach().cpu().numpy())
            
    error = error / len(loader)
    accuracy, precision, recall, f1 = compute_metrics(eval_outputs, eval_targets, task_type=task_type)
    
    if return_values:
        return [error, accuracy, precision, recall, f1, eval_targets, eval_outputs, eval_texts]
    else:
        return [error, accuracy, precision, recall, f1]


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''validation loss가 감소하면 감소를 출력한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). ')
        self.val_loss_min = val_loss
        
        
def model_freeze(model, freeze_layers=None):
    if freeze_layers == 0:
        return model
    
    if freeze_layers is not None:
        for param in model.pretrained_model.base_model.word_embedding.parameters():
            param.requires_grad = False

        if freeze_layers != -1:
            # if freeze_layer_count == -1, we only freeze the embedding layer
            # otherwise we freeze the first `freeze_layer_count` encoder layers
            for layer in model.pretrained_model.base_model.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False                 
    return model




