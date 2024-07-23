from Train_Gen.train import train
from Train_Gen.craftadv import craftadv

import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description='GAKer')

parser.add_argument('--state', type=str, default='train_model', choices=['train_model', 'craftadv','advtest'],help='Mode for model training or evaluation')
parser.add_argument('--Source_Model', type=str, default='ResNet50',help='Source Model')
parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--channel', type=int, default=32, help='Channel value')
parser.add_argument('--channel_mult', nargs='+', type=int, default=[1, 2, 3, 4], 
                    help='List of channel multipliers')
parser.add_argument('--num_res_blocks', type=int, default=1, help='Number of residual blocks')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for model training')
parser.add_argument('--Generator_save_dir', type=str, default='./GAKer_saved_checkpoint/', help='Directory to save checkpoints')
parser.add_argument('--test_load_weight', type=str, default='ckpt_19_ResNet50_.pt', help='Weight file for testing')
parser.add_argument('--set_targets', type=str, default='targets_200_cossimilar', help='target index of imagenet')
parser.add_argument('--unknown', type=str, default='False', help='if unknown or not')
parser.add_argument('--target_select', type=str, default='1', help='target_image_select')
args = parser.parse_args()
# CUDA_VISIBLE_DEVICES=2 python GAKER.py --batch_size 25 --Source_Model ResNet50 --epoch 100 --state train_model --Generator_save_dir '200cossimilar_top325_forward/'
# CUDA_VISIBLE_DEVICES=0 python ESMA.py --Source_Model ResNet50 --test_load_weight ckpt_19_ResNet50_.pt --state craftadv --Generator_save_dir './200cossimilar_top1_forward_doubleloss_eps16/' --ran_best random --set_targets targets_200_cossimilar --val_set imagenet --target_select 1


def main():

    if args.state == 'train_model':
        epoch = args.epoch
    else:
        ckpt = args.test_load_weight
        epoch = args.epoch
    modelConfig = {
        "state": args.state,
        "Source_Model": args.Source_Model,
        "epoch": epoch,
        "batch_size": args.batch_size,
        "channel": args.channel,
        "channel_mult": args.channel_mult,
        "num_res_blocks": args.num_res_blocks,
        "lr": args.lr,
        "device": args.device,
        "test_load_weight": args.test_load_weight,
        "Generator_save_dir": args.Generator_save_dir,
        'set_targets':args.set_targets,
        'unknown':args.unknown,
        'target_select':args.target_select,
    }
    
    if modelConfig["state"] == "train_model":
        train(modelConfig)
    elif modelConfig["state"] == "craftadv":
        craftadv(modelConfig)


        
if __name__ == '__main__':
    
    main()
    
    
