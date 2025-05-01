import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

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
    StyliezeBIOTClassifier
)
from utils import TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE, sharpen_prob
from utils import from_value2logits

class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model, save_path ='.log-pretrain/', test_loader = None):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_path = save_path
        self.lmda_consistency = args.lmda_consistency
        self.tau_consistency = args.tau_consistency
        self.test_data = test_loader
        # self.model.scaling_factor = 5

    def training_step(self, batch, batch_idx):
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )
        X, y = batch

        (output, f), (output_tr, f_tr) = self.model(X)
        if self.args.dataset == "CHB-MIT":
            loss_ce = focal_loss(output, y, alpha=0.8, gamma=0.7)
        else:
            loss_ce = BCE(output, y)  
        output_tr_mean = output_tr
        use_kl =  self.args.use_kl
        if use_kl:
            # loss_consistency = self.lmda_consistency * F.kl_div(F.log_softmax(output_tr_mean, dim=1),
            #                                             sharpen_prob(F.softmax(output, dim=1),
            #                                             temperature=self.tau_consistency), reduce='batchmean')
            output_tr_mean = from_value2logits(output_tr_mean)
            output = from_value2logits(output)
            loss_consistency = self.lmda_consistency * F.kl_div(output_tr_mean.log(),
                                                        sharpen_prob(output, temperature=self.tau_consistency),
                                                        reduction='batchmean')
        else:          
            pred = from_value2logits(output_tr_mean, epsilon=1e-6)
            target = from_value2logits(output,epsilon=1e-6)
            loss_consistency = 0.5 * F.binary_cross_entropy(pred, target)
        loss = loss_ce + loss_consistency

        self.log("train_loss", loss)
        self.log("train_lossce", loss_ce)
        self.log("train_loss_consistency", loss_consistency)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            # prob = self.model(X)
            (output, f), (output_tr, f_tr) = self.model(X)
            prob = output
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.validation_step_outputs.append([step_result, step_gt])
        # print(f"Validation step {batch_idx} on GPU {self.trainer.local_rank}")
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
            # print('in val')
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
            # convScore = self.model(X)
            (output, f), (output_tr, f_tr) = self.model(X)
            convScore = output
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.test_step_outputs.append([step_result, step_gt])
        return step_result, step_gt

    def on_test_epoch_end(self):
        print('in test data:self.threshold is ', self.threshold)
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
        scheduler = StepLR(optimizer, step_size=3, gamma=self.args.sche_gama)  
        return [optimizer]  , [scheduler]

def prepare_TUAB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"
    root =  "/home/yubin.he/work_dir/datasets/TUAB/edf/processed"
    root = "/data/dataset/EEG_Fundamental_Model/TUAB/v3.0.1/edf/processed"

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
    # set seed
    from pytorch_lightning import seed_everything
    number = 12345 + args.run_id
    seed_everything(number, workers=True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # get data loaders
    if args.dataset == "TUAB":
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)
    elif args.dataset == "CHB-MIT":
        train_loader, test_loader, val_loader = prepare_CHB_MIT_dataloader(args)
    else:
        raise NotImplementedError

    # define the model
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
    elif args.model == "StylizeBIOT":
        model = StyliezeBIOTClassifier(
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
    project = "binary_supervised"
    save_path = os.path.join("log-pretrain", project)
    N_version = (
        len(os.listdir(save_path)) + 1
    )
    save_path = os.path.join("log-pretrain", project, str(N_version))
    

    # logger and callbacks
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
    os.environ["WANDB_MODE"]="offline"

    logger = WandbLogger(        
        save_dir="./log-pretrain",
        project=project,
        name=f"bina-su{N_version}",
        version=f"{N_version}",
        )

    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )



    if  args.dataset == "CHB-MIT" and args.run_id >= 3 :
        # According to the experimental setup of previous work, the performance of CHB-MIT needs 
        # to be averaged over two experimental split settings in order to obtain the final results. 
        # For details, please refer to BIOT.
        temp_loader = test_loader
        test_loader = val_loader
        val_loader = temp_loader
        save_file = f'temp_{args.dataset}_log.txt'
        with open(save_file, 'a') as file:
            file.write(f'runs: {args.run_id}\n')

    trainer = pl.Trainer(
        devices=[2,3],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        # auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback],
        log_every_n_steps=50,
        gradient_clip_val=0.5,
        min_epochs=5,
        # limit_train_batches=5
    )

    lightning_model = LitModel_finetune(args, model, save_path=save_path)
    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders= val_loader
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print('best', pretrain_result)

    return pretrain_result


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=64, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset")
    parser.add_argument(
        "--model", type=str, default="StylizeBIOT", help="which supervised model to use"
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
    
    # addition paraments
    parser.add_argument("--encdata_mode", type=str, default='dwt', choices=['time_only','freq_only','both','cwt','dwt'])
    parser.add_argument("--enabale_transform", type=bool, default=False)   
    parser.add_argument("--lmda_consistency", type=float, default=10) 
    parser.add_argument("--tau_consistency", type=float, default=0.5)
    parser.add_argument("--scaling_factor", type=float, default=15) 
    parser.add_argument("--use_kl", action="store_false", help="Enable a certain feature.") #default True
    parser.add_argument("--sche_gama", type=float, default=0.1) 
    parser.add_argument("--dwt_level", type=int, default=1) 
    parser.add_argument("--style_dwtlevel", type=int, default=1) 
    parser.add_argument("--run_id", type=int, default=0)  
    args = parser.parse_args()
    print(args)
    if args.encdata_mode =='cwt' or args.encdata_mode =='dwt':
        args.enabale_transform = True
    result = supervised(args)

   

    #  best in paper in our report
    ##  TUAB
    ## python run_binary_stylize_supervised.py --encdata_mode dwt   --sche_gama 0.1 --lmda_consistency 10 --lr 1e-3 --scaling_factor 15  --dataset TUAB

    ## CHB-MIT
    ## python run_binary_stylize_supervised.py --encdata_mode dwt   --sche_gama 0.1 --lmda_consistency 10 --lr 1e-3 --scaling_factor 15 --dataset CHB-MIT


    
