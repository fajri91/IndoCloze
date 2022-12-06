#!/usr/bin/env python
# coding: utf-8

import json, glob, os, random
import argparse
import logging
import numpy as np
import pandas as pd
import torch, os
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class BertData():
    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained(args.model_type, do_lower_case=True)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.MAX_TOKEN_CHAT = args.max_token_chat
        self.MAX_TOKEN_RESP = args.max_token_resp

    def preprocess_one(self, chat, resp, label):
        chat_subtokens = [self.cls_token] + self.tokenizer.tokenize(chat) + [self.sep_token]        
        chat_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(chat_subtokens)
        if len(chat_subtoken_idxs) > self.MAX_TOKEN_CHAT:
            chat_subtoken_idxs = chat_subtoken_idxs[len(chat_subtoken_idxs)-self.MAX_TOKEN_CHAT:]
            chat_subtoken_idxs[0] = self.cls_vid

        resp_subtokens = self.tokenizer.tokenize(resp) + [self.sep_token]
        resp_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(resp_subtokens)
        if len(resp_subtoken_idxs) > self.MAX_TOKEN_RESP:
            resp_subtoken_idxs = resp_subtoken_idxs[:self.MAX_TOKEN_RESP]
            resp_subtoken_idxs[-1] = self.sep_vid

        src_subtoken_idxs = chat_subtoken_idxs + resp_subtoken_idxs
        segments_ids = [0] * len(chat_subtoken_idxs) + [1] * len(resp_subtoken_idxs)
        assert len(src_subtoken_idxs) == len(segments_ids)
        return src_subtoken_idxs, segments_ids, label
    
    def preprocess(self, chats, resps, labels):
        assert len(chats) == len(resps) == len(labels)
        output = []
        for idx in range(len(chats)):
            output.append(self.preprocess_one(chats[idx], resps[idx], labels[idx]))
        return output


class Batch():
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    # do padding here
    def __init__(self, data, idx, batch_size, device):
        PAD_ID=0
        cur_batch = data[idx:idx+batch_size]
        src = torch.tensor(self._pad([x[0] for x in cur_batch], PAD_ID))
        seg = torch.tensor(self._pad([x[1] for x in cur_batch], PAD_ID))
        label = torch.tensor([x[2] for x in cur_batch])
        mask_src = 0 + (src != PAD_ID)
        
        self.src = src.to(device)
        self.seg= seg.to(device)
        self.label = label.to(device)
        self.mask_src = mask_src.to(device)

    def get(self):
        return self.src, self.seg, self.label, self.mask_src


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.bert = BertModel.from_pretrained(args.model_type)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction='none') 

    def forward(self, src, seg, mask_src):
        batch_size = src.shape[0]
        top_vec, _ = self.bert(input_ids=src, token_type_ids=seg, attention_mask=mask_src)
        clss = top_vec[:,0,:]
        final_rep = self.dropout(clss)
        conclusion = self.linear(final_rep).squeeze()
        return self.sigmoid(conclusion)
    
    def get_loss(self, src, seg, label, mask_src):
        output = self.forward(src, seg, mask_src)
        return self.loss(output, label.float())

    def predict(self, src, seg, mask_src, label):
        output = self.forward(src, seg, mask_src)
        batch_size = output.shape[0]
        assert batch_size%2 == 0
        output = output.view(int(batch_size/2), 2)
        prediction = torch.argmax(output, dim=-1).data.cpu().numpy().tolist()
        answer = label.view(int(batch_size/2), 2)
        answer = torch.argmax(answer, dim=-1).data.cpu().numpy().tolist()
        return answer, prediction

def prediction(dataset, model, args):
    preds = []
    golds = []
    model.eval()
    assert len(dataset)%2==0
    assert args.batch_size%2==0
    for j in range(0, len(dataset), args.batch_size):
        src, seg, label, mask_src = Batch(dataset, j, args.batch_size, args.device).get()
        answer, prediction = model.predict(src, seg, mask_src, label)
        golds += answer
        preds += prediction
    return accuracy_score(golds, preds), preds

def read_data(fname):
    contexts = []
    endings = []
    labels = []
    data = pd.read_csv(fname)
    for idx, row in data.iterrows():
        context = row['Kalimat-1'] +' '+ row['Kalimat-2'] +' '+ row['Kalimat-3'] +' '+ row['Kalimat-4']
        ending1 = row['Correct Ending']
        ending2 = row['Incorrect Ending']
        
        contexts.append(context)
        endings.append(ending1)
        labels.append(1)
        
        contexts.append(context)
        endings.append(ending2)
        labels.append(0)
    return contexts, endings, labels

def read_data_en(fname):
    contexts = []
    endings = []
    labels = []
    data = pd.read_csv(fname)
    for idx, row in data.iterrows():
        context = row['InputSentence1'] +' '+ row['InputSentence2'] +' '+ row['InputSentence3'] +' '+ row['InputSentence4']
        ending1 = row['RandomFifthSentenceQuiz1']
        ending2 = row['RandomFifthSentenceQuiz2']
        if row['AnswerRightEnding'] == 1:
            contexts.append(context)
            endings.append(ending1)
            labels.append(1)
            contexts.append(context)
            endings.append(ending2)
            labels.append(0)
        else:
            assert row['AnswerRightEnding'] == 2
            contexts.append(context)
            endings.append(ending1)
            labels.append(0)
            contexts.append(context)
            endings.append(ending2)
            labels.append(1)
    return contexts, endings, labels

def train(args, train_dataset, dev_dataset, test_dataset, test2_dataset, model):
    """ Train the model """
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    t_total = len(train_dataset) // args.batch_size * args.num_train_epochs
    args.warmup_steps = int(0.1 * t_total)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warming up = %d", args.warmup_steps)
    logger.info("  Patience  = %d", args.patience)

    # Added here for reproductibility
    set_seed(args)
    tr_loss = 0.0
    global_step = 1
    best_acc_dev = 0
    best_acc_test = 0
    best_acc_test2 = 0
    cur_patience = 0
    for i in range(int(args.num_train_epochs)):
        random.shuffle(train_dataset)
        epoch_loss = 0.0
        for j in range(0, len(train_dataset), args.batch_size):
            src, seg, label, mask_src = Batch(train_dataset, j, args.batch_size, args.device).get()
            model.train()
            loss = model.get_loss(src, seg, label, mask_src)
            loss = loss.sum()/args.batch_size
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            loss.backward()

            tr_loss += loss.item()
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

        logger.info("Finish epoch = %s, loss_epoch = %s", i+1, epoch_loss/global_step)
        if i >= 4:
            dev_acc, dev_pred = prediction(dev_dataset, model, args)
            if dev_acc > best_acc_dev:
                best_acc_dev = dev_acc
                test_acc, test_pred = prediction(test_dataset, model, args)
                best_acc_test = test_acc
                test_acc2, test_pred2 = prediction(test2_dataset, model, args)
                best_acc_test2 = test_acc2
                cur_patience = 0
                logger.info("Better, BEST Acc in DEV = %s & BEST Acc in test = %s & Best Acc in test2 = %s.", best_acc_dev, best_acc_test, best_acc_test2)
            else:
                cur_patience += 1
                if cur_patience == args.patience:
                    logger.info("Early Stopping Not Better, BEST Acc in DEV = %s & BEST Acc in test = %s & Best Acc in test2 = %s.", best_acc_dev, best_acc_test, best_acc_test2)
                    break
                else:
                    logger.info("Not Better, BEST Acc in DEV = %s & BEST Acc in test = %s & BEST Acc in test2 = %s.", best_acc_dev, best_acc_test, best_acc_test2)

    return global_step, tr_loss / global_step, best_acc_dev, best_acc_test, best_acc_test2, dev_pred, test_pred, test_pred2


class Args:
    max_token_chat=450
    max_token_resp=50
    batch_size=40
    learning_rate=5e-5
    weight_decay=0
    adam_epsilon=1e-8
    max_grad_norm=1.0
    num_train_epochs=20
    warmup_steps=200
    logging_steps=200
    seed=2020
    local_rank=-1
    patience=5
    no_cuda = False
    model_type = 'bert-base-multilingual-uncased'
    
args = Args()

# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else: # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
)

args.device = device

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()


def running(training_language, test_language):
    if training_language == 'en':
        trainset = read_data_en('Data/data_en/train.csv')
        devset = read_data_en('Data/data_en/dev.csv')
    elif training_language == 'id':
        trainset = read_data('Data/data_id/train.csv')
        devset = read_data('Data/data_id/dev.csv')
    elif training_language == 'en+id':
        context_en, ending_en, label_en = read_data_en('Data/data_en/train.csv')
        context_id, ending_id, label_id = read_data_id('Data/data_id/train.csv')
        context, ending, label = context_en+context_id, ending_en+ending_id, label_en+label_id
        train_dataset = context, ending, label
        context_en, ending_en, label_en = read_data_en('Data/data_en/dev.csv')
        context_id, ending_id, label_id = read_data_id('Data/data_id/dev.csv')
        context, ending, label = context_en+context_id, ending_en+ending_id, label_en+label_id
        dev_dataset = context, ending, label
    else:
        quit()

    testset = read_data_en('Data/data_en/test.csv')
    testset2 = read_data('Data/data_id/test.csv')

    train_dataset = bertdata.preprocess(trainset[0], trainset[1], trainset[2])
    dev_dataset = bertdata.preprocess(devset[0], devset[1], devset[2])
    test_dataset = bertdata.preprocess(testset[0], testset[1], testset[2])
    test2_dataset = bertdata.preprocess(testset2[0], testset2[1], testset2[2])
    
    results = []
    for s in [1, 10, 100]:    
        bertdata = BertData(args)
        args.seed = s
        set_seed(args)

        model = Model(args, device)
        model.to(args.device)
        global_step, tr_loss, best_acc_dev, best_acc_test, best_acc_test2, dev_pred, test_pred, test_pred2 = train(args, train_dataset, dev_dataset, test_dataset, test2_dataset, model)
        print('Seed: ', args.seed)
        print('Dev set accuracy', best_acc_dev)
        print('EN Test set accuracy', best_acc_test)
        print('ID Test set accuracy', best_acc_test2)

        results.append([best_acc_dev, best_acc_test, best_acc_test2])

    print('FINAL RESULT')
    print('Dev set accuracy', np.mean([x for x,y,z in results]), np.std([x for x,y,z in results]))
    print('EN Test set accuracy', np.mean([y for x,y,z in results]), np.std([y for x,y,z in results]))
    print('ID Test set accuracy', np.mean([z for x,y,z in results]), np.std([z for x,y,z in results]))
    print('====================')

running('en')
running('id')
running('en+id')

