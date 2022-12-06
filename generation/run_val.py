import dataclasses
import json
import logging
import os
import glob
import sys
import shutil
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from utils import calculate_rouge
from run_test import get_predictions

import numpy as np
import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    DataCollator,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from data_collator import T2TDataCollator, load_data

os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger(__name__)

@dataclass
class Arguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    valid_file_path: str = field(
        metadata={"help": "Path for valid dataset"},
    )
    checkpoint_folder: str = field(
        metadata={"help": "Folder name for your current experiment"},
    )
    max_source_length: Optional[int] = field(
        default=200,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=50,
        metadata={"help": "Max input length for the target text"},
    )
    min_target_length: Optional[int] = field(
        default=5,
        metadata={"help": "Min input length for decoding"},
    )
    lang_target: str = field(
        default='en',
        metadata={"help": "either en or id"},
    )
    decoding_batch_size: Optional[int] = field(
        default=50,
        metadata={"help": "Batch size for decoding validation set"},
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )

def polish_text(text):
    special_tokens = ['[SUMMARY]', '[TITLE]', '[KEYWORD]']
    for token in special_tokens:
        text = text.replace(token, '')
    text = text.replace('<q>','\n')
    return text.strip()

def main(args_file=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((Arguments, Seq2SeqTrainingArguments)) 
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    
    # Get datasets
    logger.info('loading validation dataset')
    valid_dataset = load_data(model_args.valid_file_path)
    valid_dataset500 = random.sample(valid_dataset, 500)
    logger.info('finished loading dataset')
    
    # disable wandb console logs
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)
    results = {}
    results_rouge = {}
    for checkpoint in glob.glob(model_args.checkpoint_folder+'/checkpoint-*'):
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        # Initialize data_collator
        data_collator = T2TDataCollator(
            tokenizer=tokenizer,
            mode="training",
            using_tpu=False,
        )
        data_collator_sample = T2TDataCollator(
            tokenizer=tokenizer,
            mode="testing",
            using_tpu=False,
        )

        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
        )
        
        logger.info(f"\n*** Evaluate {checkpoint.split('/')[-1]}***")
        # Evaluation of Loss
        metrics = trainer.evaluate(
            max_length=model_args.max_target_length, num_beams=model_args.num_beams, metric_key_prefix="eval"
        )
        results [checkpoint.split('/')[-1]] = metrics
        trainer.save_metrics("eval", metrics)
    
        # Evaluation of ROUGE over smaller samples
        if model_args.lang_target == 'en':
            start_token = 'en_XX'
        elif model_args.lang_target == 'id':
            start_token = 'id_ID'

        loader = torch.utils.data.DataLoader(valid_dataset500, batch_size=model_args.decoding_batch_size, collate_fn=data_collator_sample)
        predictions, golds = get_predictions(
            model=model,
            tokenizer=tokenizer,
            data_loader=loader,
            num_beams=model_args.num_beams,
            max_length=model_args.max_target_length,
            min_length=model_args.min_target_length,
            start_token=start_token
        )
        predictions = [polish_text(pred) for pred in predictions]
        golds = [polish_text(gold) for gold in golds]
        rouge_score = calculate_rouge(predictions, golds)
        logger.info(rouge_score)
        results_rouge [checkpoint.split('/')[-1]] = rouge_score

    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]['eval_loss'])}
    logger.info('3 best checkpoints based on loss are:')
    for key in list(results.keys())[:3]:
        logger.info(key)
        logger.info(results[key])
    
    results_rouge = {k: v for k, v in sorted(results_rouge.items(), key=lambda item: item[1]['rouge1'], reverse=True)}
    logger.info('3 best checkpoints based on ROUGE-1 are:')
    for key in list(results_rouge.keys())[:3]:
        logger.info(key)
        logger.info(results_rouge[key])
    

if __name__ == "__main__":
    main()

