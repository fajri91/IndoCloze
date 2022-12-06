import dataclasses
import logging, os
import glob
from typing import Optional, Dict, Union
import numpy as np
import torch
from dataclasses import dataclass, field
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
from tqdm.auto import tqdm
from data_collator import T2TDataCollator, load_data

device = 'cuda' if torch.cuda.is_available else 'cpu'
os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger(__name__)

@dataclass
class Arguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    test_file_path: str = field(
        metadata={"help": "Path for valid dataset"},
    )
    checkpoint_folder: str = field(
        metadata={"help": "Folder name for your current experiment"},
    )
    batch_size: Optional[int] = field(
        default=30,
        metadata={"help": "Batch Size"},
    )
    max_source_length: Optional[int] = field(
        default=200,
        metadata={"help": "Max input length for the source text"},
    )
    max_decoding_length: Optional[int] = field(
        default=50,
        metadata={"help": "Max input length for the target text"},
    )
    min_decoding_length: Optional[int] = field(
        default=6,
        metadata={"help": "Min input length for the target text"},
    )
    lang_target: str = field(
        default='en',
        metadata={"help": "either en or id"},
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    output_path: Optional[str] = field(
        default="hypothesis.txt",
        metadata={"help": "path to save the generated questions."}
    )

def polish_text(text):
    text = text.replace('<pad>', '').replace('</s>', '').replace('<unk>','').replace('[QOS]', '<q>')
    return text

def get_predictions(model, tokenizer, data_loader, num_beams=10, max_length=250, min_length=50, length_penalty=1, start_token='<s>'):
    model.to(device)
    predictions = []
    golds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outs = model.generate(
                input_ids=batch['input_ids'].to(device), 
                attention_mask=batch['attention_mask'].to(device),
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                decoder_start_token_id=tokenizer.convert_tokens_to_ids(start_token)
            )
            prediction = [tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
            prediction = [polish_text(pred) for pred in prediction]
            predictions.extend(prediction)
            golds.extend([x.replace('\n', '<q>') for x in batch['tgt_txt']])
    return predictions, golds


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

    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.checkpoint_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_args.checkpoint_folder)
    logger.info("finised loading checkpoints " + model_args.checkpoint_folder.split('/')[-1])

    # Get datasets
    logger.info('loading test dataset')
    test_dataset = load_data(model_args.test_file_path)
    logger.info('finished loading dataset')
    
    # disable wandb console logs
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)
            
    collator = T2TDataCollator(
        tokenizer=tokenizer,
        mode="testing"
    )
    
    if model_args.lang_target == 'en':
        start_token = 'en_XX'
    elif model_args.lang_target == 'id':
        start_token = 'id_ID'

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=model_args.batch_size, collate_fn=collator)
    predictions, golds = get_predictions(
        model=model,
        tokenizer=tokenizer,
        data_loader=loader,
        num_beams=model_args.num_beams,
        max_length=model_args.max_decoding_length,
        min_length=model_args.min_decoding_length,
        start_token=start_token
    )
    
    checkpoint_id = model_args.checkpoint_folder.split('-')[-1]
    with open(f'{model_args.output_path}.{checkpoint_id}.candidate', 'w') as f:
        f.write("\n".join(predictions))
    with open(f'{model_args.output_path}.{checkpoint_id}.gold', 'w') as f:
        f.write("\n".join(golds))
    logging.info(f"Output saved at {model_args.output_path}.{checkpoint_id}")


if __name__ == "__main__":
    main()
    

