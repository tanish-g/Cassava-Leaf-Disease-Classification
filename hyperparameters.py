import torch
from pytorch_ranger import Ranger
from cutmix.utils import CutMixCrossEntropyLoss
class hparams():
  def params():
    return {
                "seed":42,
                "optimizer" : torch.optim.AdamW,
                "frac":0.1,
                "scheduler" : torch.optim.lr_scheduler.CosineAnnealingLR,
                "scheduler_params" : {'T_max' :20, 'eta_min':1e-6, 'last_epoch':-1, 'verbose':True},
                "fold":0,
                "lr": 2e-4,
                "weight_decay": 1e-4,
                "csv_file":"/content/fold.csv",
                "loss_fn_train":torch.nn.CrossEntropyLoss, #if cutmix is true CutMixCrossEntropyLoss(True)
                "loss_fn_val":torch.nn.CrossEntropyLoss,
                "max_epochs": 20,
                "batch_size": 64,
                "num_workers": 2,
                "image_loading_path" : "/content/train_images",
                "checkpoint": None,
                "loading_path": None,
                "saving_path":"/content/drive/",
                "Cutmix":False,
                "num_class":5,
                "beta":1.0,
                "prob":0.5,
                "num_mix":2
            }
