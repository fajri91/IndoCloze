import os, glob, json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import pandas as pd
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    model_type: Optional[str] = field(
        default="",
        metadata={"help": "Path for model"}, 
    )
    dataset_path: Optional[str] = field(
        default="",
        metadata={"help": "Path for dataset directory"}, 
    )
    train_save_to: Optional[str] = field(
        default=None,
        metadata={"help": "name for train dataset"},
    )
    dev_save_to: Optional[str] = field(
        default=None,
        metadata={"help": "name for dev dataset"},
    )
    test_save_to: Optional[str] = field(
        default=None,
        metadata={"help": "name for test dataset"},
    )
    max_source_length: Optional[int] = field(
        default=200,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=100,
        metadata={"help": "Max input length for the target text"},
    )

class DataProcessor:
    def __init__(self, model_type, max_source_length=200, max_target_length=50):
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

        self.pad_token = '<pad>'
        self.tgt_eos = '</s>'
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.lang_id_token = 'id_ID'
        self.lang_en_token = 'en_XX'

    def process(self, dataset):
        processed_dataset = []
        for datum in dataset:
            processed_data = self._convert_to_features(datum)
            if processed_data :
                processed_dataset.append(processed_data)
        return processed_dataset
  
    def _pad(self, tokens, max_length):
        while len(tokens) < max_length:
            tokens.append(self.pad_token)
        return tokens
        
    # tokenize the examples
    def _convert_to_features(self, datum):
        src = datum['src']
        tgt = datum['tgt']

        # SOURCE
        src_txt = self.tgt_eos.join(src)
        src_tokenized = [self.lang_en_token] + self.tokenizer.tokenize(src_txt)[:self.max_source_length-1]

        # SOURCE
        tgt_txt = tgt
        tgt_tokenized = [self.lang_en_token] + self.tokenizer.tokenize(tgt_txt)[:self.max_target_length-2] + [self.tgt_eos]
        
        src_tokenized = self._pad(src_tokenized, self.max_source_length)
        tgt_tokenized = self._pad(tgt_tokenized, self.max_target_length)
        
        src_idxs = torch.tensor(self.tokenizer.convert_tokens_to_ids(src_tokenized))
        tgt_idxs = torch.tensor(self.tokenizer.convert_tokens_to_ids(tgt_tokenized))
        src_mask = 0 + (src_idxs != self.pad_vid)
        
        encodings = {
            'src_idxs': src_idxs, 
            'tgt_idxs': tgt_idxs,
            'src_mask': src_mask,
            'src_txt': src_txt,
            'tgt_txt': tgt_txt,
        }
        return encodings


# Load train and dev set, by filtering
def load_mydataset(path, is_train = True):
    data = []
    if is_train:
        for idx, row in pd.read_csv(path).iterrows():
            src = [row['sentence1'], row['sentence2'], row['sentence3'], row['sentence4']]
            tgt = row['sentence5']
            data.append({
                'src': src,
                'tgt': tgt
                })
    else:
        for idx, row in pd.read_csv(path).iterrows():
            src = [row['InputSentence1'], row['InputSentence2'], row['InputSentence3'], row['InputSentence4']]
            tgt_id = row['AnswerRightEnding']
            if tgt_id == 1:
                tgt = row['RandomFifthSentenceQuiz1']
            else:
                assert tgt_id == 2
                tgt = row['RandomFifthSentenceQuiz2']
            data.append({
                'src': src,
                'tgt': tgt
                })
    return data


def main():
    parser = HfArgumentParser((DataTrainingArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    os.makedirs('/'.join(args.train_save_to.split('/')[:-1]), exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    train_dataset = load_mydataset(args.dataset_path+'/ROCStories__spring2016 - ROCStories_spring2016.csv', is_train=True)
    dev_dataset = load_mydataset(args.dataset_path+'/cloze_test_val__spring2016 - cloze_test_ALL_val.csv', is_train=False)
    test_dataset = load_mydataset(args.dataset_path+'/cloze_test_test__spring2016 - cloze_test_ALL_test.csv', is_train=False)
    
    processor = DataProcessor(args.model_type)
    
    train_dataset = processor.process(train_dataset)
    dev_dataset = processor.process(dev_dataset)
    test_dataset = processor.process(test_dataset)
 
    torch.save(train_dataset, args.train_save_to)
    logger.info(f"Train dataset, size: {len(train_dataset)}, saved at {args.train_save_to}")
    torch.save(dev_dataset, args.dev_save_to)
    logger.info(f"Dev dataset, size: {len(dev_dataset)}, saved at {args.dev_save_to}")
    torch.save(test_dataset, args.test_save_to)
    logger.info(f"Test dataset, size: {len(test_dataset)}, saved at {args.test_save_to}")

if __name__ == "__main__":
    main()

