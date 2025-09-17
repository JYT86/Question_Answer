from typing import Dict

import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer, PreTrainedTokenizer


import os

cache_dir = os.path.expanduser('/home/wangbinluo/Question_Answer/data')

def prepare_dataloaders_and_tokenizer(name='hotpotqa/hotpot_qa', batch_size=2, max_length=128, distributed=False):
    if name == 'hotpotqa/hotpot_qa':
        ds = load_dataset(name, "distractor", cache_dir=cache_dir)

    elif name == 'sentence-transformers/natural-questions':
        ds = load_dataset(name, cache_dir=cache_dir)
        
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    special_tokens = {
        'bos_token': '[BOS]',
        'eos_token': '[EOS]',
        'pad_token': '[PAD]'
    }
    tokenizer.add_special_tokens(special_tokens)
    # tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    def Tokenization(example: Dict):
        questions = ['[BOS]'+seq+'[EOS]'for seq in example['question']]
        answers = ['[BOS]'+seq+'[EOS]' for seq in example['answer']]
        # print(questions)
        questions_inputs = tokenizer(
            questions,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt'
        )
        answers_inputs = tokenizer(
            answers,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'question_input_ids': questions_inputs['input_ids'],
            'question_attn_mask': questions_inputs['attention_mask'],
            'answer_input_ids': answers_inputs['input_ids'],
            'answer_attn_mask': answers_inputs['attention_mask'],
        }
    
    if name == 'sentence-transformers/natural-questions':
        split_ds = ds['train'].train_test_split(
            test_size=0.2,
            shuffle=True,
            seed=42
        )
        ds = DatasetDict({
            'train': split_ds['train'],
            'validation': split_ds['test']
        })
        
    for split in ds:
        if name == 'sentence-transformers/natural-questions':

            def rename_query_to_question(example):
                if 'query' in example:
                    example['question'] = example['query']
                return example
            
            ds[split] = ds[split].map(
                rename_query_to_question,
                batched=True,
                num_proc=4,
                remove_columns=['query']
            ) 
        ds[split] = ds[split].map(
            Tokenization,
            batched=True,
            num_proc=4,
            remove_columns=['id', 'level', 'supporting_facts', 'context'] if name == 'hotpotqa/hotpot_qa' else None
        )
    ds = ds.with_format("torch")

    # 数据并行
    train_sampler = DistributedSampler(ds['train']) if distributed else None
    valid_sampler = DistributedSampler(ds['validation']) if distributed else None

    train_loader = DataLoader(
        ds['train'],
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler
    )
    valid_loader = DataLoader(
        ds['validation'],
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler
    )
    return {
        'train': train_loader,
        'valid': valid_loader,
        'tokenizer': tokenizer,
        'train_sampler': train_sampler,
        'valid_sampler': valid_sampler,
    }

def decode(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizer):

    decoded_text = tokenizer.decode(input_ids)
    return decoded_text

    

