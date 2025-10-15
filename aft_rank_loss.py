from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer
import torch.nn.functional as F
from transformers import GenerationConfig
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

class ScoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data):
        super(ScoreDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i])

def _single_tokenize(text, tokenizer):
    toked = tokenizer(
            text,
            return_tensors="pt"
        )

    return toked['input_ids'][0]



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        idxs = []
        all_scores = []
        input_ids = []
        labels = []
        for idx, ins in enumerate(instances):
            ins = ins['input_ids']
            query = ins['query']
            responses = ins['responses']
            scores = ins['scores']
            all_scores.append(scores)
            idxs.append([idx] * len(scores))
            #由于一个查询对应多个回复及评分，所以将 idx 重复 len(scores) 次后添加到 idxs 列表中
            query_input_ids = _single_tokenize(query, self.tokenizer)
            query_target = torch.LongTensor([IGNORE_INDEX] * query_input_ids.shape[0])
            for res in responses:
                res_input_ids = _single_tokenize(res + self.tokenizer.eos_token, self.tokenizer)  # eos here
                input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
                labels.append(torch.cat((query_target, res_input_ids), dim=0))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, train_data, mix=False) -> Dict:

    train_dataset = ScoreDataset(data=train_data)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, data_collator=data_collator)

