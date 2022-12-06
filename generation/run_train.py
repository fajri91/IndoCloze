import dataclasses
import json
import logging
import os
import sys
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from transformers.trainer_utils import get_last_checkpoint

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
    train_file_path: str = field(
        metadata={"help": "Path for train dataset"},
    )
    valid_file_path: str = field(
        metadata={"help": "Path for valid dataset"},
    )
    model_path: str = field(
        metadata={"help": "Path for pretrained model"},
    )
    experiment_path: str = field(
        metadata={"help": "Path for your experiment output"},
    )
    experiment_name: str = field(
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
    lang_target: str = field(
        default='en',
        metadata={"help": "either en or id"},
    )
    max_train_samples: Optional[int] = field(
        default=False,
        metadata={"help": "For debugging, cut some data"},
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )

def check_and_create(logger, experiment_path, experiment_name, overwrite=False):
    exp_path = experiment_path+'/'+experiment_name
    if not os.path.exists(experiment_path):
        logger.info('Experiment_path doesn"t exist, will create it')
        os.mkdir(experiment_path)
    if os.path.exists(exp_path):
        if not overwrite:
            raise Exception('Experiment name exists, please choose other name')
        shutil.rmtree(experiment_path+'/'+experiment_name)
    os.mkdir(experiment_path+'/'+experiment_name)

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
    
    # Create experiment folder
    if training_args.do_train and training_args.resume_from_checkpoint is None:
        check_and_create(logger, model_args.experiment_path, model_args.experiment_name, overwrite=True)
    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    config = AutoConfig.from_pretrained(model_args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_path, config=config)

    model.resize_token_embeddings(len(tokenizer))

    # Get datasets
    logger.info('loading dataset')
    
    train_dataset = load_data(model_args.train_file_path)
    valid_dataset = load_data(model_args.valid_file_path)

    logger.info('finished loading dataset')

    # Initialize data_collator
    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        mode="training",
        using_tpu=False,
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    # disable wandb console logs
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)

    # Training
    if training_args.do_train:
        output_dir = model_args.experiment_path + '/' + model_args.experiment_name
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            model_args.max_train_samples if model_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()
