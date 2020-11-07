import numpy as np
import pandas as pd
import string
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


class NerGritDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'I-PERSON': 0, 'B-ORGANISATION': 1, 'I-ORGANISATION': 2, 'B-PLACE': 3, 'I-PLACE': 4, 'O': 5, 'B-PERSON': 6}
    INDEX2LABEL = {0: 'I-PERSON', 1: 'B-ORGANISATION', 2: 'I-ORGANISATION', 3: 'B-PLACE', 4: 'I-PLACE', 5: 'O', 6: 'B-PERSON'}
    NUM_LABELS = 7
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data) 
    
    
class NerKtpDataset(Dataset):
    # Static constant variable
    LABELS = [
        'U-FLD_PROV', 'B-VAL_PROV', 'L-VAL_PROV', 'U-FLD_KAB', 'U-VAL_KAB',
        'U-FLD_NIK', 'U-VAL_NIK', 'U-FLD_NAMA', 'B-VAL_NAMA', 'L-VAL_NAMA',
        'B-FLD_TTL', 'L-FLD_TTL', 'B-VAL_TTL', 'L-VAL_TTL', 'B-FLD_GDR',
        'L-FLD_GDR', 'U-VAL_GDR', 'B-FLD_GLD', 'L-FLD_GLD', 'U-VAL_GLD',
        'U-FLD_ADR', 'B-VAL_ADR', 'I-VAL_ADR', 'L-VAL_ADR', 'U-FLD_RTW',
        'U-VAL_RTW', 'U-FLD_KLH', 'U-VAL_KLH', 'U-FLD_KCM', 'U-VAL_KCM',
        'U-FLD_RLG', 'U-VAL_RLG', 'B-FLD_KWN', 'L-FLD_KWN', 'B-VAL_KWN',
        'L-VAL_KWN', 'U-FLD_KRJ', 'U-VAL_KRJ', 'U-FLD_WRG', 'U-VAL_WRG',
        'B-FLD_BLK', 'L-FLD_BLK', 'B-VAL_BLK', 'L-VAL_BLK', 'U-VAL_SGP',
        'U-VAL_SGD', 'B-VAL_KAB', 'L-VAL_KAB', 'U-VAL_NAMA', 'B-VAL_KLH',
        'L-VAL_KLH', 'B-VAL_KRJ', 'I-VAL_KRJ', 'L-VAL_KRJ', 'B-VAL_SGP',
        'L-VAL_SGP', 'I-VAL_TTL', 'L-VAL_KCM', 'B-VAL_KCM', 'U-VAL_KWN',
        'U-VAL_PROV', 'I-VAL_NAMA', 'I-VAL_PROV', 'I-VAL_KAB', 'I-VAL_KCM',
        'I-VAL_SGP', 'U-VAL_ADR', 'I-VAL_KLH', 'O'
    ]
    
    LABEL2INDEX = dict((label,idx) for idx, label in enumerate(LABELS))
    INDEX2LABEL = dict((idx, label) for idx, label in enumerate(LABELS))
    NUM_LABELS = len(LABELS)
    
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def load_dataset(self, path):
        dframe = pd.read_csv(path)
        
        dataset, sentence, seq_label = [], [], []
        length_sentence = len(dframe.sentence_idx.unique())
        for idx in tqdm(range(length_sentence)):
            sframe = dframe[dframe.sentence_idx == idx]
            for sidx in range(len(sframe)):
                line = sframe.iloc[sidx]
                word = str(line.word)
                label = str(line.tag)
                sentence.append(word)
                seq_label.append(self.LABEL2INDEX[label])
            dataset.append({
                'sentence': sentence,
                'seq_label': seq_label
            })
            sentence, seq_label = [], []
        return dataset
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
        
class NerDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NerDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)
        
        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list