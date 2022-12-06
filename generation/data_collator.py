from typing import Dict, List, Optional

import torch

def trim_batch(input_ids, pad_token_id, attention_mask=None,):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def load_data(path):
    dataset = torch.load(path)
    final_dataset = []
    for data in dataset:
        final_dataset.append({
            'src_idxs': data['src_idxs'],
            'src_mask': data['src_mask'],
            'tgt_idxs': data['tgt_idxs'],
            'tgt_txt': data['tgt_txt'],
            })
    return final_dataset

# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessary because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
class T2TDataCollator():
    def __init__(self, tokenizer, mode='training', using_tpu=False) :
        self.tokenizer = tokenizer
        self.mode = mode
        self.using_tpu = using_tpu

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = torch.stack([example['src_idxs'] for example in batch])
        attention_mask = torch.stack([example['src_mask'] for example in batch])
        
        # don't trim on tpu, for some reason trimming leads to slower training on TPU
        if not self.using_tpu:
            input_ids, attention_mask = trim_batch(input_ids, pad_token_id, attention_mask=attention_mask)
        
        params =  {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
        }
        
        if self.mode == 'training':
            target_ids = torch.stack([example['tgt_idxs'] for example in batch])
            target_ids = trim_batch(target_ids, pad_token_id)
            if not self.using_tpu:
                target_ids = trim_batch(target_ids, pad_token_id)
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
            lm_labels[lm_labels == pad_token_id] = -100

            params["labels"] = lm_labels
            params["decoder_input_ids"] = decoder_input_ids
        
        if self.mode == 'testing':
            params['tgt_txt'] = [example['tgt_txt'] for example in batch]

        return params
