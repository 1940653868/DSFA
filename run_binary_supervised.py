import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pyhealth.metrics import binary_metrics_fn

from model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
)
from utils import TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model, save_path ='.log-pretrain/'):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_path = save_path

    def training_step(self, batch, batch_idx):
        # if self.global_step % 2000 == 0:
        #     self.trainer.save_checkpoint(
        #         filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
        #     )
        X, y = batch
        prob = self.model(X)
        if self.args.dataset == "CHB-MIT":
            loss = focal_loss(prob, y, alpha=0.8, gamma=0.7)
        else:
            loss = BCE(prob, y)  # 
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.validation_step_outputs.append([step_result, step_gt])
        return step_result, step_gt

    def on_validation_epoch_end(self):
        result = np.array([])
        gt = np.array([])
        val_step_outputs = self.validation_step_outputs
        for out in val_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])

        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["balanced_accuracy","pr_auc", "roc_auc", "accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("val_auroc", result["roc_auc"], sync_dist=True)
        self.print('in val data:', result)
        self.validation_step_outputs.clear() 

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.test_step_outputs.append([step_result, step_gt])
        return step_result, step_gt

    def on_test_epoch_end(self):
        result = np.array([])
        gt = np.array([])
        test_step_outputs = self.test_step_outputs
        for out in test_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("test_auroc", result["roc_auc"], sync_dist=True)
        self.print('in test data, result is:', result)
        self.test_step_outputs.clear()
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]

def prepare_TUAB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"
    root =  "/home/yubin.he/work_dir/datasets/TUAB/edf/processed"

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    try:
        enabale_transform = args.enabale_transform
    except:
        enabale_transform = False
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate, enabale_transform=enabale_transform, args =  args),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate, enabale_transform=enabale_transform, args =  args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate, enabale_transform = enabale_transform, args =  args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader



def prepare_CHB_MIT_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/physionet.org/files/chbmit/1.0.0/clean_segments"
    root = "/home/yubin.he/work_dir/datasets/CHB-MIT/physionet.org/files/chbmit/1.0.0/clean_segments" 
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    try:
        enabale_transform = args.enabale_transform
    except:
        enabale_transform = False
    train_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "train"),
                     train_files, args.sampling_rate, enabale_transform=enabale_transform, args =  args),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "test"),
                     test_files, args.sampling_rate, enabale_transform=enabale_transform, args =  args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "val"),
                    val_files, args.sampling_rate,enabale_transform=enabale_transform, args =  args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_PTB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/WFDB/processed2"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def supervised(args):
    # get data loaders
    if args.dataset == "TUAB":
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)
    elif args.dataset == "CHB-MIT":
        train_loader, test_loader, val_loader = prepare_CHB_MIT_dataloader(args)
    else:
        raise NotImplementedError

    # define the model
    # ["SPaRCNet",  "ContraWR", "CNNTransformer", "FFCL","STTransformer", "BIOT"]
    if args.model == "SPaRCNet":
        model = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sampling_rate * args.sample_length),
            n_classes=args.n_classes,
            block_layers=4,
            growth_rate=16,
            bn_size=16,
            drop_rate=0.5,
            conv_bias=True,
            batch_norm=True,
        )

    elif args.model == "ContraWR":
        model = ContraWR(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
        )

    elif args.model == "CNNTransformer":
        model = CNNTransformer(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.sampling_rate,
            steps=args.hop_length // 5,
            dropout=0.2,
            nhead=4,
            emb_size=256,
        )

    elif args.model == "FFCL":
        model = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sampling_rate * args.sample_length),
            shrink_steps=20,
        )

    elif args.model == "STTransformer":
        model = STTransformer(
            emb_size=256,
            depth=4,
            n_classes=args.n_classes,
            channel_legnth=int(
                args.sampling_rate * args.sample_length
            ),  # (sampling_rate * duration)
            n_channels=args.in_channels,
        )

    elif args.model == "BIOT":
        model = BIOTClassifier(
            n_classes=args.n_classes,
            # set the n_channels according to the pretrained model if necessary
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
            args = args
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    else:
        raise NotImplementedError
    
    project = "binary_supervised_Baseline"
    save_path = os.path.join("log-pretrain", project)
    N_version = (
        len(os.listdir(save_path)) + 1
    )
    save_path = os.path.join("log-pretrain", project, str(N_version))
    lightning_model = LitModel_finetune(args, model, save_path=save_path)

    # logger and callbacks
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
    os.environ["WANDB_MODE"]="offline"


    logger = WandbLogger(        
        save_dir="./wandb",
        project = project,
        version=f"{version}",
        name=f"log-sud{N_version}-{args.model}",
        )

    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=30, verbose=False, mode="max"
    )
    class TestEveryNCallback(pl.Callback):
        def __init__(self, every_n_epochs: int, trianer=None):
            self.every_n_epochs = every_n_epochs
            self.current_epoch = 0
            # self.trainer = trainer

        def on_epoch_end(self, trainer, pl_module):
            if (self.current_epoch + 1) % self.every_n_epochs == 0:
                results = trainer.test(model=pl_module, ckpt_path="best",dataloaders=test_loader)
                print(f"Epoch {self.current_epoch+1} Test Results: {results}")
            self.current_epoch += 1
            print('now we go into end of epoch')

    # 创建回调实例并添加到Trainer
    test_callback = TestEveryNCallback(every_n_epochs=5)  # 每5个epoch测试一次
    trainer = pl.Trainer(
        devices=[0,1],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        # auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback,test_callback],
        log_every_n_steps=10
    )
    # val_test_group = [val_loader,test_loader]
    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders = val_loader #val_loader
    )

    # test the model
    
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders = test_loader #test_loader #test_loader
    )[0]
    print(pretrain_result)


if __name__ == "__main__":
    from main_binary_supervised import set_args_func
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=64, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset") 
    parser.add_argument(
        "--model", type=str, default="BIOT", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=16, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="number of output classes"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="", help="pretrained model path"
    )
    # encdata_mode = 'cwt'
    parser.add_argument("--encdata_mode", type=str, default='freq_only', choices=['time_only','freq_only','both','cwt', 'dwt'])
    parser.add_argument("--enabale_transform", type=bool, default=False)
    args = parser.parse_args()
    # args = set_args_func(return_parser=False)
    print(args)
    if args.encdata_mode =='cwt' or 'dwt':
        args.enabale_transform = True
    supervised(args)
    # ["SPaRCNet",  "ContraWR", "CNNTransformer", "FFCL","STTransformer", "BIOT"]
    # python run_binary_supervised.py   --encdata_mode freq_only --enabale_transform False
    # python run_binary_supervised.py --dataset CHB-MIT --batch_size 512 --model SPaRCNet
    # python run_binary_supervised.py --dataset CHB-MIT --batch_size 512 --model ContraWR
    # python run_binary_supervised.py --dataset CHB-MIT --batch_size 512 --model CNNTransformer
    # python run_binary_supervised.py --dataset CHB-MIT --batch_size 512 --model FFCL
    # python run_binary_supervised.py --dataset CHB-MIT --batch_size 512 --model STTransformer
    # python run_binary_supervised.py --dataset CHB-MIT --batch_size 512 --model BIOT
    
    # # python run_binary_supervised.py --dataset CHB-MIT --batch_size 512 --model BIOT
