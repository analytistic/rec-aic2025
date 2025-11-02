from .exp import BasicExp
from dataprocess.dataset import RecDataset, ItemDataset
import pandas as pd
from dataprocess.sample import BucketSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import SASRec
from utils import Top1Rec
from functools import partial



class Infer(BasicExp):
    def __init__(self, cfg):
        self.cfg = cfg

        self.test_dataset = self._get_dataset('test')
        self.item_dataset = ItemDataset(pd.read_csv(cfg.dataset.item_features_path), self.test_dataset.item_col_feat_dict, self.test_dataset.book_id2re_id)

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_sampler=BucketSampler(self.test_dataset, self.cfg.dataset.bs * self.cfg.dataset.len, bucket_size=self.cfg.dataset.bucket_size, shuffle=False),
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=partial(self.test_dataset.collate_fn, flag='test'),
            pin_memory=False,
        )
        self.item_dataset_loader = DataLoader(
            self.item_dataset,
            batch_size=self.cfg.dataset.item_loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.item_loader.num_workers,
            collate_fn=self.item_dataset.collate_fn,
            pin_memory=False,
        )
        self.model = self._build_model()


    def _get_dataset(self, flag='test'):
        interactions = pd.read_csv(self.cfg.dataset.interactions_path)
        user_features = pd.read_csv(self.cfg.dataset.user_features_path)
        item_features = pd.read_csv(self.cfg.dataset.item_features_path)
        return RecDataset(self.cfg, interactions, user_features, item_features, flag=flag)




    def _build_model(self):
        model_name = self.cfg.model.name
        map_model = {
            'sasrec': SASRec,
        }
        return map_model[model_name](cfg=self.cfg, user_id_num=self.test_dataset.user_id_num, item_id_num=self.test_dataset.item_id_num, user_feat_dict=self.test_dataset.user_feat_dict, item_feat_dict=self.test_dataset.item_feat_dict)
    
    def predict(self, topk: int=1):
        user_id_list = []
        log_vector = []
        item_vector = []
        top1_list = []
        self.model.load_state_dict(torch.load(self.cfg.test.checkpoint_path))
        self.model.eval()
        book_id = []
        top1_index = []



        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
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
                ) = batch


                log_seq, user_id = self.model.predict(
                    user_id,
                    j,
                    user_feat,
                    id_seq,
                    feat_seq,
                    inter_time,
                    act_type,
                    token_type,
                    renew_seq,
                )
                log_vector.append(log_seq)
                user_id_list.extend(user_id.cpu().numpy().tolist())

            for i, batch in tqdm(enumerate(self.item_dataset_loader), total=len(self.item_dataset_loader)):
                item_id, re_id, item_feat = batch
                item_emb = self.model.save_item_emb(item_id, item_feat)
                item_vector.append(item_emb)
                book_id.extend(item_id.cpu().numpy().tolist())

            item_vector = torch.cat(item_vector, dim=0) # (item_num, dim)

            for i in range(len(log_vector)):
                if log_vector[i] is None:
                    continue
                log_vector[i] = log_vector[i]
                scores = torch.matmul(log_vector[i], item_vector.T)
                _, top1 = scores.topk(topk, dim=-1)
                top1_index.extend(top1.squeeze(-1).cpu().numpy().tolist())

        topk1_list = [book_id[i] for i in top1_index]

        top1_rec = Top1Rec(user_id_list=user_id_list, top1_item_list=topk1_list)
        return top1_rec
