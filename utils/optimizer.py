"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import torch 
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer

BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased', 
}

# SBERT_CLASS = {
#     "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
#     "minilm6": "all-MiniLM-L6-v2"
# }

SBERT_CLASS = {     # fastest: paraphrase-MiniLM-L3-v2
    "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
    "minilm-v2": "all-MiniLM-L6-v2",
    "multi-mini": "multi-qa-MiniLM-L6-cos-v1",
    "mpnet-v2": "all-mpnet-base-v2",
    "multi-mpnet": "multi-qa-mpnet-base-dot-v1",
    "paraph-minilm": "paraphrase-MiniLM-L3-v2",
    "multi-distilbert": "multi-qa-distilbert-cos-v1",
    "glove": "average_word_embeddings_glove.6B.300d",
    "multi-mpnet-50": "paraphrase-multilingual-mpnet-base-v2",
    "bert": "bert-base-nli-mean-tokens"
}


def get_optimizer(model, args):
    
    optimizer = torch.optim.Adam([
        {'params':model.bert.parameters()}, 
        {'params':model.contrast_head.parameters(), 'lr': args.lr * args.lr_scale},
        {'params':model.cluster_centers, 'lr': args.lr * args.lr_scale}
    ], lr=args.lr)
    
    print(optimizer)
    return optimizer 
    

def get_bert(args):
    
    if args.use_pretrain == "SBERT":
        path = SBERT_CLASS[args.bert]
        bert_model = get_sbert(args)
        tokenizer = bert_model[0].tokenizer
        model = bert_model[0].auto_model
        print("..... loading Sentence-BERT !!!")
    else:
        path = BERT_CLASS[args.bert]
        config = AutoConfig.from_pretrained(path)
        model = AutoModel.from_pretrained(path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
        print("..... loading plain BERT !!!")
        
    return model, tokenizer, path


def get_sbert(args):
    sbert = SentenceTransformer(SBERT_CLASS[args.bert], device="cuda")
    return sbert








