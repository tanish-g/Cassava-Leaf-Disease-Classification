import sys
from PIL import Image
import torch
import pandas as pd
import os
import torch.nn.functional as F
from fastprogress.fastprogress import master_bar, progress_bar
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from augmentation import get_augmentations
import numpy as np
import random
from model import leafEfficientNet
from dataset import leafDataset
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import warnings
warnings.simplefilter('ignore')
from cutmix.cutmix import CutMix
from hyperparameters import hparams
from get_train_val_split import split

hyparam=hparams.params()
train_tfms = get_augmentations(img_size=128).train_tfms()
test_tfms = get_augmentations(img_size=128).test_tfms()

df=pd.read_csv(hyparam['csv_file'])
train_df,valid_df=split(df).get_train_val_split()
train_df=train_df.sample(frac=hyparam['frac']).reset_index(drop=True)
valid_df=valid_df.sample(frac=hyparam['frac']).reset_index(drop=True)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(hyparam['seed'])

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_model(model_name='efficientnet-b0',lr=3e-3,wd=1e-4,opt_fn=hyparam['optimizer'],device=None,checkpoint=hyparam['checkpoint'],path=hyparam['loading_path']):
    device = device if device else get_device()
    model = leafEfficientNet(model_name=model_name)
    opt = opt_fn(model.parameters(),lr=lr,weight_decay=wd)
    model = model.to(device)
    if checkpoint:
        checkpoint_path=torch.load(path)
        model.load_state_dict(checkpoint_path['model_state_dict'])
        opt.load_state_dict(checkpoint_path['optimizer_state_dict'])
        del checkpoint_path
    return model, opt

def training_step(xb,yb,model,loss_fn,opt,device,scheduler):
    xb,yb = xb.to(device), yb.to(device)
    out = model(xb)
    opt.zero_grad()
    loss = loss_fn()(out,yb)
    loss.backward()
    opt.step()
    scheduler.step()
    return loss.item()
    
def validation_step(xb,yb,model,loss_fn,device):
    xb,yb = xb.to(device), yb.to(device)
    out = model(xb)
    loss = loss_fn()(out,yb)
    _, out = torch.max(out, 1)
    return loss.item(),out

def get_data(train_df,valid_df,train_tfms,test_tfms):
    train_ds = leafDataset(df=train_df,im_path=hyparam['image_loading_path'],transforms=train_tfms)
    if hyparam['Cutmix']==True:
      train_ds = CutMix(train_ds, num_class=hyparam['num_class'], beta=hyparam['beta'], prob=hyparam['prob'], num_mix=hyparam['num_mix'])
    train_dl = DataLoader(dataset=train_ds,batch_size=hyparam['batch_size'],shuffle=True,num_workers=hyparam['num_workers'])
    valid_ds = leafDataset(df=valid_df,im_path=hyparam['image_loading_path'],transforms=test_tfms)
    valid_dl = DataLoader(dataset=valid_ds,batch_size=hyparam['batch_size'],shuffle=False,num_workers=hyparam['num_workers'])
    return train_dl,valid_dl

def fit(epochs=hyparam['max_epochs'],model=None,train_dl=None,valid_dl=None,opt=None,device=None,loss_fn_train=hyparam['loss_fn_train'],loss_fn_val=hyparam['loss_fn_val'],checkpoint=hyparam['checkpoint'],loading_path=hyparam['loading_path'],saving_path=hyparam['saving_path']):
    device = device if device else get_device()
    scheduler = hyparam['scheduler'](opt,T_max=hyparam['scheduler_params']['T_max'],eta_min=hyparam['scheduler_params']['eta_min'])
    if checkpoint:
        checkpoint_path=torch.load(path)
        scheduler.load_state_dict(checkpoint_path['scheduler_state_dict'])
        del checkpoint_path
    best_acc=0
    mb = master_bar(range(epochs))
    for epoch in mb:    
        trn_loss,val_loss = 0.0,0.0
        val_preds=[]
        val_targets=[]
        ##Training
        model.train()
        for xb,yb in progress_bar(train_dl,parent=mb):
            trn_loss += training_step(xb,yb,model,loss_fn_train,opt,device,scheduler)
        trn_loss /= mb.child.total
        ##Validation
        model.eval()
        with torch.no_grad():
            for i,(xb,yb) in enumerate(progress_bar(valid_dl,parent=mb)):
                loss,out = validation_step(xb,yb,model,loss_fn_val,device)
                val_loss += loss
                val_preds.extend(out.detach().cpu().numpy())
                val_targets.extend(yb.detach().cpu().numpy())
        val_loss /= mb.child.total
        val_acc = accuracy_score(val_targets,val_preds)
        scheduler.step(val_acc)
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(model.state_dict(),f'model.pth')
        torch.save({'model_state_dict':model.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'optimizer_state_dict':opt.state_dict(),'val_loss':val_loss},f'model_last.pth')
        print(f'Epoch: {epoch},Train_loss: {trn_loss:.5f},Val_loss:{val_loss:.5f} Val_accuracy:{val_acc:.4f}')
        with open(f'log.txt','a+') as f:
            f.writelines(f'Epoch: {epoch},Train_loss: {trn_loss:.5f} Val_loss:{val_loss:.5f} Val_accuracy:{val_acc:.4f}\n')
    import yagmail
    sendtome = ['tanishgupta34@gmail.com']
    contents = open(f'log.txt').readlines()
    yag = yagmail.SMTP(user = 'labsc202@gmail.com' , password = 'labelsmooth1')
    yag.send(to = sendtome , subject = 'Knock Knock Model has trained !!',contents = contents ,attachments=['log.txt'])
    del train_dl,valid_dl,scheduler,opt
    return model

train_dl,valid_dl = get_data(train_df,valid_df,train_tfms,test_tfms)
model,opt = get_model(model_name=hyparam['model_name'],lr=hyparam['lr'],wd=hyparam['weight_decay'])

model = fit(model=model,train_dl=train_dl,valid_dl=valid_dl,opt=opt)
