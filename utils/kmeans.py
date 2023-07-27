"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
import wandb as wb

insert_keyword = lambda dic, word: dict([(f"{word}/{k}", v) for k, v in dic.items()])
format_float = lambda dic, ndigit: dict([(k, round(v, ndigit)) for k, v in dic.items()])

def get_mean_embeddings(bert, input_ids, attention_mask, token_type_ids=None):
        bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    
def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return token_feat


def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length, keyword):
    for i, batch in enumerate(train_loader):

        text, label = batch['text'], batch['label']
        tokenized_features = get_batch_token(tokenizer, text, max_length).to("cuda")
        # for tensor in tokenized_features.values():
            #  tensor = tensor.to("cuda")
              
        corpus_embeddings = get_mean_embeddings(bert, **tokenized_features)
        
        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings.detach().cpu().numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().cpu().numpy()), axis=0)

    # Perform KMeans clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes, n_init=20, random_state=wb.config.seed) # I changed this TODO
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)    
    print("{}: all_embeddings:{}, true_labels:{}, pred_labels:{}".format(keyword, all_embeddings.shape, len(true_labels), len(pred_labels)))

    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    km_scores = confusion.clusterscores()
    print("{}, Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(keyword, clustering_model.n_iter_, confusion.acc(), clustering_model.cluster_centers_.shape))
    
    worded_km_scores = insert_keyword(dic=km_scores, word=keyword)
    return clustering_model.cluster_centers_, worded_km_scores



