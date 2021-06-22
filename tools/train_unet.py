import os
import time
from argparse import ArgumentParser

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import optim
from torch.nn import SyncBatchNorm, CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from unet_mod.models.unet import TinyUNet
from unet_mod.datasets.mod import MOD
from unet_mod.utils import collect_env, get_root_logger, ClosedFormCosineLRScheduler, accuarcy, mIoU


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--root", default='/data/datasets/mod_dataset')
    parser.add_argument(
        "--train_annfile", default='/data/datasets/mod_dataset/train_list.txt')
    parser.add_argument(
        "--val_annfile", default='/data/datasets/mod_dataset/val_list.txt')
    parser.add_argument("--samples_per_gpu", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--use_fp16", action='store_true')
    parser.add_argument("--clip_grad_norm", type=float)
    parser.add_argument("--log_dir", default='exps/unet')
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = arg_parser()
    # turn on benchmark mode
    torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(fp16=args.use_fp16)

    if accelerator.is_main_process:
        # setup logger
        os.makedirs(args.log_dir, exist_ok=True)
        time_stamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        logger = get_root_logger(logger_name='MOD', log_file=os.path.join(
                args.log_dir, f'{time_stamp}.log'))
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tf_logs'))
        # log env info
        logger.info('--------------------Env info--------------------')
        for key, value in sorted(collect_env().items()):
            logger.info(str(key) + ': ' + str(value))
        # log args
        logger.info('----------------------Args-----------------------')
        for key, value in sorted(vars(args).items()):
            logger.info(str(key) + ': ' + str(value))
        logger.info('---------------------------------------------------')

    train_dataset = MOD(root=args.root, annfile=args.train_annfile)
    train_dataloader = DataLoader(train_dataset, batch_size=args.samples_per_gpu,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
    ## val dataloader
    val_dataset = MOD(root=args.root, annfile=args.val_annfile, val=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.samples_per_gpu,
                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # define model
    model = TinyUNet(n_channels=1, n_classes=train_dataset.num_classes, bilinear=True)
    # optimizer
    init_lr = args.base_lr*dist.get_world_size()*args.samples_per_gpu/16
    optimizer = optim.SGD(model.parameters(), lr=init_lr,
                          weight_decay=1e-4, momentum=0.9)
    # recover states
    start_epoch = 1
    if args.resume is not None:
        ckpt: dict() = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']+1
        if accelerator.is_main_process:
            logger.info(f"Resume from epoch {start_epoch-1}...")
    else:
        if accelerator.is_main_process:
            logger.info("Start training from scratch...")
    # convert BatchNorm to SyncBatchNorm
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    # prepare to be DDP models
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader)
    # closed_form lr_scheduler
    total_steps = len(train_dataloader)*args.epochs
    resume_step = len(train_dataloader)*(start_epoch-1)
    lr_scheduler = ClosedFormCosineLRScheduler(optimizer, init_lr, total_steps, resume_step)
    ## loss criterion
    criterion = CrossEntropyLoss(weight=torch.tensor([1., 5.]), ignore_index=255).to(accelerator.device) # 
    # training
    ## Best acc
    best_miou = 0.
    for e in range(start_epoch, args.epochs+1):
        model.train()
        for i, batch in enumerate(train_dataloader):
            img, mask = batch
            logits = model(img)
            loss = criterion(logits, mask)
            accelerator.backward(loss)
            # clip grad if true
            if args.clip_grad_norm is not None:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            # sync before logging
            accelerator.wait_for_everyone()
            ## log and tensorboard
            if accelerator.is_main_process:
                if i % args.log_interval == 0:
                    writer.add_scalar('loss', loss.item(),
                                        (e-1)*len(train_dataloader)+i)
                    lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('lr', lr,
                                        (e-1)*len(train_dataloader)+i)
                    loss_str = f"loss: {loss.item():.4f}"
                    epoch_iter_str = f"Epoch: [{e}] [{i}/{len(train_dataloader)}], "
                    if args.clip_grad_norm is not None:
                        logger.info(
                            epoch_iter_str+f'lr: {lr}, '+loss_str+f', grad_norm: {grad_norm}')
                    else:
                        logger.info(epoch_iter_str+f'lr: {lr}, '+loss_str)

            lr_scheduler.step()
        if accelerator.is_main_process:
            if e % args.save_interval == 0:
                save_path = os.path.join(args.log_dir, f'epoch_{e}.pth')
                torch.save(
                    {'state_dict': model.module.state_dict(), 'epoch': e, 'args': args,
                        'optimizer': optimizer.state_dict()}, save_path)
                logger.info(f"Checkpoint has been saved at {save_path}")
        ## start to evaluate
        if accelerator.is_main_process:
            logger.info("Evaluate on validation dataset")
            bar = tqdm(total=len(val_dataloader))
        model.eval()
        preds = []
        gts = []
        for batch in val_dataloader:
            img, mask = batch
            with torch.no_grad():
                logits = model(img)
                pred = F.softmax(logits, dim=1)
                pred = torch.argmax(pred, dim=1)
                pred = accelerator.gather(pred)
                gt = accelerator.gather(mask)
            preds.append(pred)
            gts.append(gt)
            if accelerator.is_main_process:
                bar.update(accelerator.num_processes)
        if accelerator.is_main_process:
            bar.close()
            ## compute metrics
            preds = torch.cat(preds)[:len(val_dataloader.dataset)]
            gts = torch.cat(gts)[:len(val_dataloader.dataset)]
            # accuarcy
            acc = accuarcy(preds, gts, ignore_index=0, average='micro')
            # mIoU
            miou = mIoU(preds, gts, ignore_index=0)
            logger.info(f"Accuracy on Val dataset: {acc:.4f}")
            logger.info(f"Mean IoU on Val dataset: {miou:.4f}")
            ## save preds
            if miou>best_miou:
                best_miou = miou
                val_results_dir = os.path.join(args.log_dir, 'best_val_results')
                os.makedirs(val_results_dir, exist_ok=True)
                imgpaths = val_dataset.imgpaths
                assert preds.shape[0]==len(imgpaths)
                preds = preds.cpu().numpy()
                for i in range(preds.shape[0]):
                    imgname = imgpaths[i].split('/')[-1]
                    imgpath = os.path.join(val_results_dir, imgname)
                    result = preds[i].astype(np.uint8)
                    result[result==1] = 255
                    result = Image.fromarray(result)
                    result.save(imgpath)
        ## delete unuseful vars
        del preds
        del gts
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
