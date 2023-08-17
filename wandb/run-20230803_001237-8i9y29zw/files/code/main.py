"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
from pathlib import Path
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
from models.Transformers import SCCLBert
import dataloader.dataloader as dataloader
from training import SCCLvTrainer
from utils.kmeans import get_kmeans_centers
from utils.logger import setup_path, set_global_random_seed
from utils.optimizer import get_optimizer, get_bert
import numpy as np
import wandb as wb

def run(args):
    print(args, end="\n\n")
    with wb.init(project="sccl-2021", mode=args.log_mode, group=args.dataname, config=args) as current_run:
        current_run.tags =[args.dataname, args.bert, str(args.eta)]
        current_run.name = "|".join(current_run.tags) + "|" + current_run.id
        args.resPath, args.tensorboard = setup_path(args)
        set_global_random_seed(args.seed)
        
        # dataset loader
        train_loader = dataloader.explict_augmentation_loader(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader(args)

        # model
        torch.cuda.set_device(args.gpuid[0])
        bert, tokenizer = get_bert(args)
        bert.to("cuda")
        # TODO: assert cuda device 
        assert next(bert.parameters()).device.type == "cuda"
        # initialize cluster centers
        cluster_centers, first_base_scores = get_kmeans_centers(bert, tokenizer, train_loader, args.num_classes, args.max_length, keyword="first-base")
        wb.run.summary.update(first_base_scores)
        print("first-base kmeans repre scores:", first_base_scores)

        model = SCCLBert(bert, tokenizer, cluster_centers=cluster_centers, alpha=args.alpha) 
        #if True:
        #
        #    checkpoint = r"D:\\text_clustering_paper\\my-forks\sccl_fork\\models\saved_models\searchsnippets\\ldlehh3y-minilm6-best-model.pth"
        #    model.load_state_dict(state_dict=torch.load(f=Path(checkpoint).__str__()))
        model = model.cuda()
        assert next(model.parameters()).device.type == "cuda"
        assert next(model.contrast_head.parameters()).device.type == "cuda"
        assert model.cluster_centers.device.type == "cuda"


        # optimizer 
        optimizer = get_optimizer(model, args)
        trainer = SCCLvTrainer(model, tokenizer, optimizer, train_loader, args)
        loss_dic, repre_scores, model_scores = trainer.train()
        _ , last_base_scores = get_kmeans_centers(bert, tokenizer, train_loader, args.num_classes, args.max_length, keyword="last-base")
        wb.run.summary.update(last_base_scores)
        print("##### the final kmeans repre scores: ", last_base_scores)
        
    wb.run.finish()
    return None


def get_my_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local') 
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=42, help="")
    parser.add_argument('--print_freq', type=float, default=25, help="")
    parser.add_argument('--resdir', type=str, default='results/')
    parser.add_argument('--s3_resdir', type=str, default='results')
    parser.add_argument('--log_mode', default="offline")
    parser.add_argument('--bert', type=str, default='minilm6', help="")
    parser.add_argument('--use_pretrain', type=str, default='SBERT', choices=["BERT", "SBERT", "PAIRSUPCON"])
    
    # Dataset
    parser.add_argument('--datapath', type=str, default='datasets/')
    parser.add_argument('--dataname', type=str, default='stackoverflow', help="")
    parser.add_argument('--num_classes', type=int, default=20, help="")
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--label', type=str, default='label')
    parser.add_argument('--text', type=str, default='text')
    parser.add_argument('--augmentation_1', type=str, default='text1')
    parser.add_argument('--augmentation_2', type=str, default='text2')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=1000)
    # contrastive learning
    parser.add_argument('--objective', type=str, default='SCCL')
    parser.add_argument('--augtype', type=str, default='virtual', choices=['virtual', 'explicit'])
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=10, help="")
    parser.add_argument('--alpha', type=float, default=1.0)

    # --temperature 0.5 \
    #     --eta 10 \
    #     --lr 1e-05 \
    #     --lr_scale 100 \
    #     --max_length 32 \
    #     --batch_size 400 \
    #     --max_iter 1000 \
    #     --print_freq 100 \
    #     --gpuid 1 &
    
    # Clustering
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None
    
    if args.dataname == "searchsnippets":
        args.num_classes = 8
        args.alpha = 1.
    elif args.dataname == "biomedical":
        args.num_classes = 20
        args.alpha = 10.
    elif args.dataname == "stackoverflow":
        args.num_classes = 20
        args.alpha = 1.
    else:
        print("##### unknown dataset...")
        return None

    return args
"""
python3 main.py \
        --resdir $path-to-store-your-results \
        --use_pretrain SBERT \
        --bert distilbert \
        --datapath $path-to-your-data \
        --dataname searchsnippets \
        --num_classes 8 \
        --text text \
        --label label \
        --objective SCCL \
        --augtype virtual \
        --temperature 0.5 \
        --eta 10 \
        --lr 1e-05 \
        --lr_scale 100 \
        --max_length 32 \
        --batch_size 400 \
        --max_iter 1000 \
        --print_freq 100 \
        --gpuid 1 &
"""
if __name__ == '__main__':
    import subprocess
       
    args = get_my_args(sys.argv[1:])
    
    
    

    if args.train_instance == "sagemaker":
        run(args)
        subprocess.run(["aws", "s3", "cp", "--recursive", args.resdir, args.s3_resdir])
    else:
        run(args)
            



    











"""
def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local') 
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=100, help="")
    parser.add_argument('--resdir', type=str, default='./results/')
    parser.add_argument('--s3_resdir', type=str, default='./results')
    
    parser.add_argument('--bert', type=str, default='distilroberta', help="")
    parser.add_argument('--use_pretrain', type=str, default='BERT', choices=["BERT", "SBERT", "PAIRSUPCON"])
    
    # Dataset
    parser.add_argument('--datapath', type=str, default='../datasets/')
    parser.add_argument('--dataname', type=str, default='stackoverflow', help="")
    parser.add_argument('--num_classes', type=int, default=20, help="")
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--label', type=str, default='label')
    parser.add_argument('--text', type=str, default='text')
    parser.add_argument('--augmentation_1', type=str, default='text1')
    parser.add_argument('--augmentation_2', type=str, default='text2')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=1000)
    # contrastive learning
    parser.add_argument('--objective', type=str, default='contrastive')
    parser.add_argument('--augtype', type=str, default='virtual', choices=['virtual', 'explicit'])
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=1, help="")
    
    # Clustering
    parser.add_argument('--alpha', type=float, default=1.0)
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args
"""