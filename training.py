"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import time
import numpy as np
from sklearn import cluster
import wandb as wb
from time import time
from pathlib import Path
from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader
import copy


import torch
import torch.nn as nn
import time
from torch.nn import functional as F
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss

insert_keyword = lambda dic, word: {f"{word}/{k}": v                for k, v in dic.items()}
format_float = lambda dic, ndigit: {k            : round(v, ndigit) for k, v in dic.items()}
import functools

global_step = 0

def time_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        dif = time.time() - start

        # dif = round((end - start)/60, ndigits=2)
        if global_step < 20 or global_step in [0, 50, 100]:
            print(f'$$$ timer {global_step} $$$ /{func.__name__}/ ==> {dif/60:.2f} MIN')
        if global_step % 20 == 0:
            wb.log({f't(m)/{func.__name__}': dif/60, f't(s)/{func.__name__}': dif})

        return results
    return wrapper
    


class SCCLvTrainer(nn.Module): 
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        print("training with eta : ", self.eta)
        self.save_model_path = Path(args.save_dir).resolve() / self.args.dataname / wb.run.id / "checkpoints" 
        # self.save_best_path = Path(args.save_dir).resolve() / self.args.dataname / wb.run.id/ "bests" 
        self.save_embed_path = Path(args.save_dir).resolve() / self.args.dataname / wb.run.id / "embeds" 
        self.save_model_path.mkdir(parents=True, exist_ok=True)
        self.save_embed_path.mkdir(parents=True, exist_ok=True)
        print(self.save_model_path.__str__(), self.save_embed_path.__str__(), sep='\n')
        wb.run.summary["save_model_path"] = self.save_model_path.__str__()
        wb.run.summary["save_embed_path"] = self.save_embed_path.__str__()

        self.best_model_scores = None
        self.num_best_updates = 0
        self.cluster_loss = nn.KLDivLoss(reduction="sum")
        self.contrast_loss = PairConLoss(temperature=self.args.temperature)
        
        self.gstep = 0
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")
    
    def update_best_model(self, current_model):
        # print(current_model.keys())
        path = None
        if self.best_model_scores is None or current_model["model/avg"] > self.best_model_scores["best/model/avg"]:
            # self.best_model_scores = copy.deepcopy(current_model)
            self.best_model_scores =  {f"best/{k}": v for k, v in current_model.items()}
            
            wb.run.summary.update(self.best_model_scores)
            wb.run.log(self.best_model_scores)
            self.num_best_updates += 1
            path = self.save_model_path / f"best_model.tar"
            state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': global_step}
            torch.save(obj=state, f=path.__str__())
            print("/\/\/\ new BEST /\/\/\ ")
        return path
            # self.model()

    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        
    @time_func
    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)

            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
            
        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)
            
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            
        return input_ids.cuda(), attention_mask.cuda()
        
    @time_func
    def train_step_virtual(self, input_ids, attention_mask, itr):
        
        embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        losses = self.contrast_loss(feat1, feat2)
        wb.log({"constrast-raw": losses["loss"].item()})
        
        # contrastive_loss_value = losses["loss"].item()
        loss = self.eta * losses["loss"]
        contrastive_loss_value = loss.item()
        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target) / output.shape[0]
            wb.log({"cluster-raw": cluster_loss.item()})
            cluster_loss *= 0.5
            loss += cluster_loss
            cluster_loss_value = cluster_loss.item()
            
            losses["cluster_loss"] = cluster_loss_value
        
        total_loss_value = loss.item()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        wb_dic = {
            "loss/iter":       itr,
            "loss/total":          total_loss_value,
            "loss/cont_scaled": contrastive_loss_value,
            "loss/clus_scaled":      cluster_loss_value,
            "loss/sum-const-clust": contrastive_loss_value + cluster_loss_value,
            "loss/contrast-ratio":  contrastive_loss_value / total_loss_value,
            "loss/cluster-ratio":       cluster_loss_value / total_loss_value,

            }

        return losses, wb_dic
    
    
    @time_func
    def train(self):
        print("#" * 40)
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))
        
        repre_scores, model_scores = self.evaluate_embedding(-1)
        wb.run.log(repre_scores)
        wb.run.log(model_scores)
        best_path = self.update_best_model(model_scores)
       
        wb.watch(self.model)
        self.model.train()
        # t0 = time()
        for i in np.arange(self.args.max_iter+1):
            global global_step
            global_step += 1
            
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch)



            losses, loss_dic = self.train_step_virtual(input_ids, attention_mask, itr=i) if self.args.augtype == "virtual" else self.train_step_explicit(input_ids, attention_mask)
            wb.run.log(loss_dic)
            
            #(self.args.print_freq > 0) and 
            if (i != 0) and (self.args.print_freq > 0) and ((i%self.args.print_freq == 0)  or (i == self.args.max_iter)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                repre_scores, model_scores = self.evaluate_embedding(i)
                wb.run.log(repre_scores)
                wb.run.log(model_scores)
                best_path = self.update_best_model(model_scores)


            
                self.model.train()
            
            if i % self.args.check_freq == 0:
                path = self.save_model_path / f"checkpoint_iter_{i}.tar"
                state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': i}
                torch.save(obj=state, f=path.__str__())
            
        print("best model saved at", best_path)
        wb.run.summary["num-best-updates"] = self.num_best_updates
        return loss_dic, repre_scores, model_scores   

    @time_func
    def evaluate_embedding(self, step):
        dataloader = unshuffle_loader(self.args)
        # print('---- {} evaluation batches ----'.format(len(dataloader)))
        
        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label'] 
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")

                model_prob = self.model.get_cluster_prob(embeddings)
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)
                    
        # Initialize confusion matrices
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)
        
        all_pred = all_prob.max(1)[1]
        confusion_model.add(all_pred, all_labels)
        confusion_model.optimal_assignment(self.args.num_classes)
        acc_model = confusion_model.acc()

        kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed, n_init=20)
        embeddings = all_embeddings.cpu().numpy()
        kmeans.fit(embeddings)
        pred_labels = torch.tensor(kmeans.labels_.astype(int))
        
        # clustering accuracy 
        confusion.add(pred_labels, all_labels)
        confusion.optimal_assignment(self.args.num_classes)
        acc = confusion.acc()

        ressave = {"acc":acc, "acc_model":acc_model}
        ressave.update(confusion.clusterscores())
        # for key, val in ressave.items():
            # self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)
            
        # np.save(self.args.resPath + 'acc_{}.npy'.format(step), ressave)
        # np.save(self.args.resPath + 'scores_{}.npy'.format(step), confusion.clusterscores())
        # np.save(self.args.resPath + 'mscores_{}.npy'.format(step), confusion_model.clusterscores())
        # np.save(self.args.resPath + 'mpredlabels_{}.npy'.format(step), all_pred.cpu().numpy())
        # np.save(self.args.resPath + 'predlabels_{}.npy'.format(step), pred_labels.cpu().numpy())
        
        embed_path= self.save_embed_path  / f"embeds__iter_{step}.npy"
        label_path = self.save_embed_path / f"labels__iter_{step}.npy"
        # torch.save(obj=self.model.state_dict(), f=path.__str__())
        np.save(embed_path, embeddings)
        np.save(label_path, all_labels.cpu().numpy())

        repre_scores = confusion.clusterscores()
        model_scores = confusion_model.clusterscores()
        repre_scores_rounded = format_float(dic=repre_scores, ndigit=2)
        model_scores_rounded = format_float(dic=model_scores, ndigit=2)
        
        print(f'Iter {step}: [Representation]  scores: ', repre_scores_rounded) 
        # print('[Representation] ACC: {:.3f}'.format(acc)) 
        print(f'Iter {step}:          [Model]  scores: ', model_scores_rounded) 
        # print('[Model] ACC: {:.3f}'.format(acc_model))

        repre_scores_2 = insert_keyword(dic=repre_scores, word="repre")
        model_scores_2 = insert_keyword(dic=model_scores, word="model")
        repre_scores_2["iter"] = step
        model_scores_2["iter"] = step
        return repre_scores_2, model_scores_2



    
    def train_step_explicit(self, input_ids, attention_mask):
        
        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            loss += cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    