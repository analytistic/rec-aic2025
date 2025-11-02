import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .sample import BucketSampler
from collections import defaultdict


class RecDataset(Dataset):
    def __init__(self, cfg, interactions: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame, flag: str = 'train'):
        self.cfg = cfg
        self.interactions = interactions
        self.user_features = user_features
        self.item_features = item_features
        self.flag = flag
        self.user_feat_dict = {}
        self.item_feat_dict = {}
        self.item_col_feat_dict = {}
        self.book_id2re_id = defaultdict(int)
        self._preprocess()
        self.item_id_set = set(self.item_features['book_id'].unique())
        self.user_seq_lens = self.interactions.groupby('user_id').size().to_dict()
        # self.item_id_num = len(self.book_id2re_id)
    




    def _preprocess(self):
        for col in self.user_features.columns:
            if col != '借阅人':
                feat = self.user_features[col].unique().tolist()
                feat_dict = {v: (i+1) for i, v in enumerate(feat)}
                self.user_features[col] = self.user_features[col].map(feat_dict)
                self.user_features[col].fillna(0, inplace=True)
                self.user_feat_dict[f'{col}'] = len(feat)
                self.user_id_num = len(self.user_features['借阅人'].unique())
        for col in self.item_features.columns:
            if col not in ['book_id', '题名']:
                feat = self.item_features[col].unique().tolist()
                feat_dict = {v: (i+1) for i, v in enumerate(feat)}
                self.item_features[col] = self.item_features[col].map(feat_dict)
                self.item_features[col].fillna(0, inplace=True)
                self.item_feat_dict[f'{col}'] = len(feat)
                self.item_id_num = len(self.item_features['book_id'].unique())
                self.item_col_feat_dict[f'{col}'] = feat_dict


        self.interactions['还书时间'] = self.interactions['还书时间'].astype(str).str.replace(r'(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})', r'\1 \2', regex=True)
        self.interactions['续借时间'] = self.interactions['续借时间'].astype(str).str.replace(r'(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})', r'\1 \2', regex=True)
        # self.interactions['借阅时间'] = self.interactions['借阅时间'].astype(str).str.replace(r'(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})', r'\1 \2', regex=True)
        self.interactions['借阅时间'] = pd.to_datetime(self.interactions['借阅时间'])
        self.interactions['还书时间'] = pd.to_datetime(self.interactions['还书时间'])
        # self.interactions['续借时间'] = pd.to_datetime(self.interactions['续借时间'])
        self.interactions = self.interactions.sort_values(by='借阅时间').reset_index(drop=True)
        self.maxrenew = self.interactions['续借次数'].max()
        book_borrow_counts = self.interactions['book_id'].value_counts().sort_values(ascending=False)
        reid = 1
        for book_id, count in book_borrow_counts.items():
            if count > 1:
                self.book_id2re_id[book_id] = reid
                reid += 1

        if self.flag == 'train':
            if isinstance(self.cfg.dataset.train_ratio, float):
                self.interactions = self.interactions[:int(len(self.interactions) * self.cfg.dataset.train_ratio)]
            else:
                self.interactions = (
                    self.interactions
                    .sort_values(['user_id', '借阅时间'])
                    .groupby('user_id')
                    .apply(lambda df: df.iloc[:-1])
                    .reset_index(drop=True)
                )

        elif self.flag == 'val':
            pass
        elif self.flag == 'test':
            pass
        else:
            raise ValueError('flag must be train, val or test')
        

        self.interactions_user_ids = self.interactions['user_id'].unique().tolist()

        borrow_df = self.interactions.copy()
        return_df = self.interactions.copy()
        borrow_df['inter_time'] = borrow_df['借阅时间']
        return_df['inter_time'] = return_df['还书时间']
        borrow_df['act_type'] = 1  # 借阅
        return_df['act_type'] = 2 # 归还
        interactions = pd.concat([borrow_df, return_df], ignore_index=True)
        self.interactions = interactions[['inter_id', 'user_id', 'book_id', 'inter_time', 'act_type', '续借次数']]
        self.interactions = self.interactions.sort_values(by=['user_id', 'inter_time']).reset_index(drop=True)
        self.interactions['inter_time'] = self.interactions['inter_time'].astype(np.int64) // 10**9 



        
            
        
    def __len__(self):
        return len(self.interactions_user_ids)
    

    def _getitem_fortrainer(self, idx):
        user_id = self.interactions_user_ids[idx]
        inter_seq = self.interactions[self.interactions['user_id'] == user_id].reset_index(drop=True)
        length = len(inter_seq)
        id_seq = np.zeros(length, dtype=np.int32)
        reid_seq = np.zeros(length, dtype=np.int32)
        feat_seq = np.zeros((length, len(self.item_feat_dict)), dtype=np.int32)
        user_feat = np.zeros(len(self.user_feat_dict), dtype=np.int32)
        pos_seq = np.zeros(length, dtype=np.int32)
        reid_pos_seq = np.zeros(length, dtype=np.int32)
        reid_neg_seq = np.zeros(length, dtype=np.int32)
        pos_feat = np.zeros((length, len(self.item_feat_dict)), dtype=np.int32)
        act_type = np.zeros(length, dtype=np.int32)
        token_type = np.zeros(length, dtype=np.int32)
        inter_time = np.zeros(length, dtype=np.int64)
        renew_seq = np.zeros(length, dtype=np.int32)

        last_borrow_idx = inter_seq[inter_seq['act_type'] == 1].index.max()
        next_borrow = inter_seq.loc[last_borrow_idx, 'book_id']
        # 从最后一个开始填充
        j = length - 1
        for i in reversed(range(last_borrow_idx)):
            row = inter_seq.iloc[i]
            act = row['act_type']
            book_id = row['book_id']
            ts = row['inter_time']
            renew = row['续借次数']
            pos_seq[j] = next_borrow
            id_seq[j] = book_id
            act_type[j] = act
            inter_time[j] = ts
            token_type[j] = 1
            renew_seq[j] = renew
            reid_seq[j] = self.book_id2re_id.get(book_id, 0)
            reid_pos_seq[j] = self.book_id2re_id.get(next_borrow, 0)
            if act == 1:
                next_borrow = book_id
            else:
                pass
            j -= 1

        feat_seq = self.item_features.set_index('book_id').reindex(id_seq)[list(self.item_feat_dict.keys())].fillna(0).astype(np.int32).values
        pos_feat = self.item_features.set_index('book_id').reindex(pos_seq)[list(self.item_feat_dict.keys())].fillna(0).astype(np.int32).values
        # 负采样
        neg_seq = np.zeros(length, dtype=np.int32)
        id_set = set(inter_seq['book_id'].values)
        neg_candidates = np.array(list(self.item_id_set - id_set))
        neg_seq = np.random.choice(neg_candidates, size=length, replace=True)
        for i in range(length):
            reid_neg = self.book_id2re_id.get(neg_seq[i], 0)
        neg_feat = self.item_features.set_index('book_id').reindex(neg_seq)[list(self.item_feat_dict.keys())].fillna(0).astype(np.int32).values

        user_feat = self.user_features[self.user_features['借阅人'] == user_id].iloc[0][list(self.user_feat_dict.keys())].fillna(0).astype(np.int32).values
        token_type[j] = 2

        return user_id, j, user_feat, id_seq, reid_seq, feat_seq, pos_seq, reid_pos_seq, pos_feat, neg_seq, reid_neg_seq, neg_feat, inter_time, act_type, token_type, renew_seq

    def _getitem_fortest(self, idx):
        user_id = self.interactions_user_ids[idx]
        inter_seq = self.interactions[self.interactions['user_id'] == user_id].reset_index(drop=True)
        length = len(inter_seq) + 1
        id_seq = np.zeros(length, dtype=np.int32)
        reid_seq = np.zeros(length, dtype=np.int32)
        feat_seq = np.zeros((length, len(self.item_feat_dict)), dtype=np.int32)
        user_feat = np.zeros(len(self.user_feat_dict), dtype=np.int32)
        act_type = np.zeros(length, dtype=np.int32)
        token_type = np.zeros(length, dtype=np.int32)
        inter_time = np.zeros(length, dtype=np.int64)
        renew_seq = np.zeros(length, dtype=np.int32)


        j = len(inter_seq) - 1
        for i in reversed(range(length)):
            row = inter_seq.iloc[j]
            act = row['act_type']
            book_id = row['book_id']
            ts = row['inter_time']
            renew = row['续借次数']
            id_seq[i] = book_id
            act_type[i] = act
            inter_time[i] = ts
            token_type[i] = 1
            renew_seq[i] = renew
            reid_seq[i] = self.book_id2re_id.get(book_id, 0)
            j -= 1
            if j < 0:
                break


        feat_seq = self.item_features.set_index('book_id').reindex(id_seq)[list(self.item_feat_dict.keys())].fillna(0).astype(np.int32).values
        user_feat = self.user_features[self.user_features['借阅人'] == user_id].iloc[0][list(self.user_feat_dict.keys())].fillna(0).astype(np.int32).values
        token_type[0] = 2
        j = 0

        return user_id, j, user_feat, id_seq, reid_seq, feat_seq, inter_time, act_type, token_type, renew_seq



    def __getitem__(self, idx):
        if self.flag == 'train':
            return self._getitem_fortrainer(idx)
        elif self.flag == 'val':
            return self._getitem_fortrainer(idx)
        elif self.flag == 'test':
            return self._getitem_fortest(idx)
        else:
            raise ValueError('flag must be train, val or test')

    @staticmethod
    def collate_fn(batch, flag = 'train'):
        def pad_seqs(seqs):
            return pad_sequence([torch.tensor(seq) for seq in seqs], batch_first=True, padding_value=0, padding_side= 'left')
        if flag == 'train' or flag == 'val':
            (
                user_id,
                j,
                user_feat,
                id_seq,
                reid_seq,
                feat_seq,
                pos_seq,
                reid_pos_seq,
                pos_feat,
                neg_seq,
                reid_neg_seq,
                neg_feat,
                inter_time,
                act_type,
                token_type,
                renew_seq,
            ) = zip(*batch)
            seq_lens = torch.tensor([len(seq) for seq in id_seq])

            id_seq = pad_seqs(id_seq)
            reid_seq = pad_seqs(reid_seq)
            feat_seq = pad_seqs(feat_seq)
            pos_seq = pad_seqs(pos_seq)
            reid_pos_seq = pad_seqs(reid_pos_seq)
            pos_feat = pad_seqs(pos_feat)
            neg_seq = pad_seqs(neg_seq)
            reid_neg_seq = pad_seqs(reid_neg_seq)
            neg_feat = pad_seqs(neg_feat)
            inter_time = pad_seqs(inter_time)
            act_type = pad_seqs(act_type)
            token_type = pad_seqs(token_type)
            renew_seq = pad_seqs(renew_seq)
            user_feat = torch.tensor(user_feat, dtype=torch.long)
            user_id = torch.tensor(user_id, dtype=torch.long)
            j = torch.tensor(j, dtype=torch.long) + (id_seq.size(1) - seq_lens)

            return (
                user_id,
                j,
                user_feat,
                id_seq,
                reid_seq,
                feat_seq,
                pos_seq,
                reid_pos_seq,
                pos_feat,
                neg_seq,
                reid_neg_seq,
                neg_feat,
                inter_time,
                act_type,
                token_type,
                renew_seq,
            )
        elif flag == 'test':
            (
                user_id,
                j,
                user_feat,
                id_seq,
                reid_seq,
                feat_seq,
                inter_time,
                act_type,
                token_type,
                renew_seq,
            ) = zip(*batch)
            seq_lens = torch.tensor([len(seq) for seq in id_seq])

            id_seq = pad_seqs(id_seq)
            reid_seq = pad_seqs(reid_seq)
            feat_seq = pad_seqs(feat_seq)
            inter_time = pad_seqs(inter_time)
            act_type = pad_seqs(act_type)
            token_type = pad_seqs(token_type)
            renew_seq = pad_seqs(renew_seq)
            user_feat = torch.tensor(user_feat, dtype=torch.long)
            user_id = torch.tensor(user_id, dtype=torch.long)
            j = torch.tensor(j, dtype=torch.long) + (id_seq.size(1) - seq_lens)

            return (
                user_id,
                j,
                user_feat,
                id_seq,
                reid_seq,
                feat_seq,
                inter_time,
                act_type,
                token_type,
                renew_seq,
            )
        else:
            raise ValueError('flag must be train, val or test')
    

class ItemDataset(Dataset):
    def __init__(self, item_features: pd.DataFrame, item_col_feat_dict, book_id2re_id):
        self.item_features = item_features
        self.item_col_feat_dict = item_col_feat_dict
        self.book_id2re_id = book_id2re_id
        self.item_ids = item_features['book_id'].unique().tolist()
        self._preprocess()

    def _preprocess(self):
        for col in self.item_features.columns:
            if col not in ['book_id', '题名']:
                self.item_features[col] = self.item_features[col].map(self.item_col_feat_dict[f'{col}'])
                self.item_features[col].fillna(0, inplace=True)


    def __len__(self):
        return len(self.item_ids)


    def __getitem__(self, idx):
        item_id = self.item_ids[idx]
        feat = self.item_features[self.item_features['book_id'] == item_id].iloc[0][list(self.item_col_feat_dict.keys())].astype(np.int32).values
        re_id = self.book_id2re_id.get(item_id, 0)
        return item_id, re_id, feat
    

    @staticmethod
    def collate_fn(batch):
        item_ids, re_id, feats = zip(*batch)
        item_ids = torch.tensor(item_ids)
        re_id = torch.tensor(re_id)
        feats = torch.tensor(feats)
        return item_ids, re_id, feats

        
        









