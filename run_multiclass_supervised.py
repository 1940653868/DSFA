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
from pyhealth.metrics import multiclass_metrics_fn

from model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
    StyliezeBIOTClassifier
)
from utils import TUEVLoader, HARLoader


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model, save_path ='.log-multi-super/'):
        super().__init__()
        self.args = args
        self.model = model
        self.args = args
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_path = save_path


    def training_step(self, batch, batch_idx):
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )
        X, y = batch
        prod = self.model(X)
        loss = nn.CrossEntropyLoss()(prod, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            # step_result = torch.argmax(convScore, dim=1).cpu().numpy()
            step_result = convScore.cpu().numpy()
            step_gt = y.cpu().numpy()
        # print(max(step_gt), max(np.argmax(step_result, axis=1)))
        # print(min(step_gt), min(np.argmax(step_result, axis=1)))
        self.validation_step_outputs.append([step_result, step_gt])
        return step_result, step_gt

    def on_validation_epoch_end(self):
        result = []
        gt = np.array([])
        val_step_outputs = self.validation_step_outputs
        for out in val_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])

        result = np.concatenate(result, axis=0)
        result = multiclass_metrics_fn(
            gt, result, metrics=["accuracy","balanced_accuracy","cohen_kappa", "f1_weighted"]
        )
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("val_f1", result["f1_weighted"], sync_dist=True)
        self.log("val_balanced_accuracy", result["balanced_accuracy"], sync_dist=True)
        self.print('in val data', result)
        self.validation_step_outputs.clear() 

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = convScore.cpu().numpy()
            # step_result = torch.argmax(convScore, dim=1).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.test_step_outputs.append([step_result, step_gt])
        return step_result, step_gt

    def on_test_epoch_end(self):
        result = []
        gt = np.array([])
        test_step_outputs = self.test_step_outputs
        for out in test_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])

        result = np.concatenate(result, axis=0)
        result = multiclass_metrics_fn(
            gt, result, metrics=["accuracy", "cohen_kappa", "f1_weighted","balanced_accuracy"]
        )
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("test_f1", result["f1_weighted"], sync_dist=True)
        self.log("test_balanced_accuracy", result["balanced_accuracy"], sync_dist=True)
        self.print('in test data :', result)
        self.test_step_outputs.clear()
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]


def prepare_TUEV_dataloader(args):
    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    pl.seed_everything(seed=seed)
    root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"
    root = "/home/yubin.he/work_dir/datasets/TUEV/v2.0.1/edf"
    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]
    try:
        enabale_transform = args.enabale_transform
    except:
        enabale_transform = False
    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, args.sampling_rate, enabale_transform, encdata_mode =  args.encdata_mode
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, args.sampling_rate, enabale_transform, encdata_mode =  args.encdata_mode
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, args.sampling_rate, enabale_transform, encdata_mode =  args.encdata_mode
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_HAR_dataloader(args):
    # set random seed
    
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/HAR/processed/"

    train_files = os.listdir(os.path.join(root, "train"))
    test_files = os.listdir(os.path.join(root, "test"))
    val_files = os.listdir(os.path.join(root, "val"))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        HARLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def supervised(args):
    # get data loaders
    if args.dataset == "TUEV":
        train_loader, test_loader, val_loader = prepare_TUEV_dataloader(args)

    else:
        raise NotImplementedError

    # define the model
    if args.model == "SPaRCNet":
        model = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sample_length * args.sampling_rate),
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
            n_segments=4 if args.dataset == "HAR" else 5,
        )

    elif args.model == "FFCL":
        model = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sample_length * args.sampling_rate),
            shrink_steps=16 if args.dataset == "HAR" else 20,
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
            from collections import OrderedDict

            checkpoint = torch.load(args.pretrain_model_path)
            state_dict = checkpoint['state_dict']

            # 修改 state_dict 的键，使其与模型定义匹配
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace('model.biot.', '')  # 修改键名的规则视具体情况而定
                new_state_dict[new_key] = v

            model.biot.load_state_dict(new_state_dict, strict=False)#
            # model.biot.load_state_dict(torch.load(args.pretrain_model_path)['state_dict'])
            print(f"load pretrain model from {args.pretrain_model_path}")


    else:
        raise NotImplementedError
    

    project = "multi-supervised_TUEV" 
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
        project= project,   
        name=f"multi-su{N_version}",
        save_dir="./log-pretrain",
        version=f"{N_version}",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_cohen", patience=5, verbose=False, mode="max"
    )
    if args.pretrain_model_path:
        min_epochs = 5
        # min_epochs = 35
        devices = [int(it) for it in args.devices]
    else:
        min_epochs = 5
        if args.devices:
            devices = [int(it) for it in args.devices]
        else:
            devices = [6]

    trainer = pl.Trainer(
        devices=devices,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        # auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback],
        log_every_n_steps=25,
        # min_epochs=15  #open it when train from scratch #close it when train from checkpoints
    )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=test_loader # val
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print(pretrain_result)
    return pretrain_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUEV", help="dataset")
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
        "--n_classes", type=int, default=6, help="number of output classes"
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
    # encdata_mode = 'cwt' 前面三轮次cwt跑来微调效果远超于freq
    parser.add_argument("--encdata_mode", type=str, default='freq_only', choices=['time_only','freq_only','both','cwt','dwt'])
    parser.add_argument("--enabale_transform", type=bool, default=False)    
    parser.add_argument("--lmda_consistency", type=float, default=0.5) 
    parser.add_argument("--tau_consistency", type=float, default=0.5) 
    parser.add_argument("--scaling_factor", type=float, default=15) 
    parser.add_argument('--devices', nargs='+', default=[6],help='Pass in a list of values')
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    print(args)
    if args.encdata_mode =='cwt' or args.encdata_mode =='dwt':
        args.enabale_transform = True
    # supervised(args)

    get_five_run = True
    results = []
    if get_five_run:
        for i in range(args.runs):
            result = supervised(args)
            results.append(result)
        # {'test_acc': 0.7734112739562988, 'test_cohen': 0.7720161080360413, 'test_f1': 0.8514195680618286, 'test_balanced_accuracy': 0.8572990894317627}
        acc_list = [item['test_acc'] for item in results]
        cohen_list = [item['test_cohen'] for item in results]
        test_f1_list = [item['test_f1'] for item in results]
        b_acc_list = [item['test_balanced_accuracy'] for item in results]
        from utils import calculate_mean_std_dev
        acc_all = calculate_mean_std_dev(acc_list)
        cohen_all = calculate_mean_std_dev(cohen_list)
        b_acc = calculate_mean_std_dev(b_acc_list)
        test_f1_all = calculate_mean_std_dev(test_f1_list)
        print(f'acc: {acc_all}')
        print(f'cohen: {cohen_all}')
        print(f'b_acc: {b_acc}')
        print(f'test_f1: {test_f1_all}')
        
        with open('finetune_tuev.txt', 'a') as file:
            file.write(f'3data{args.pretrain_model_path}\n')
            file.write(f'acc: {acc_all}\n')
            file.write(f'cohen: {cohen_all}\n')
            file.write(f'b_acc: {b_acc}\n')
            file.write(f'test_f1: {test_f1_all}\n')
            
            file.write('============================================================\n')
        
    else:
        result = supervised(args)
    # train from scratch
    # python run_multiclass_supervised.py  --encdata_mode freq_only

    # pretrain setting
    # python run_multiclass_supervised.py --pretrain_model_path pretrain_model/IIIC.ckpt --encdata_mode freq_only
    # python run_multiclass_supervised.py --pretrain_model_path pretrain_model/CHB-MIT.ckpt --encdata_mode freq_only
    # python run_multiclass_supervised.py --pretrain_model_path pretrain_model/TUAB.ckpt --encdata_mode freq_only
    
    
    # su3da
    # python run_multiclass_supervised.py --pretrain_model_path pretrain_model/su3_99.ckpt --encdata_mode freq_only --devices 0
    # python run_multiclass_supervised.py --pretrain_model_path pretrain_model/su3_51.ckpt --encdata_mode freq_only --devices 1
    # python run_multiclass_supervised.py --pretrain_model_path pretrain_model/su3_12.ckpt --encdata_mode freq_only --devices 2