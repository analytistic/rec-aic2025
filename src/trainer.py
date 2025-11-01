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



class Trainer(BasicExp):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg)
        self.cfg = cfg
        self.train_dataset = self._get_dataset('train')
        self.valid_dataset = self._get_dataset('val')
        self.item_dataset = ItemDataset(pd.read_csv(cfg.dataset.item_features_path), self.train_dataset.item_col_feat_dict)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=BucketSampler(self.train_dataset, self.cfg.dataset.bs * self.cfg.dataset.len, bucket_size=self.cfg.dataset.bucket_size, shuffle=True),
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=False,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_sampler=BucketSampler(self.valid_dataset, self.cfg.dataset.bs * self.cfg.dataset.len, bucket_size=self.cfg.dataset.bucket_size, shuffle=False),
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=self.valid_dataset.collate_fn,
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.learning_rate)
        self.writer = SummaryWriter(log_dir=self.cfg.log_dir)
        
    def _build_model(self):
        model_name = self.cfg.model.name
        map_model = {
            'sasrec': SASRec,
        }
        return map_model[model_name](cfg=self.cfg, user_id_num=self.train_dataset.user_id_num, item_id_num=self.train_dataset.item_id_num, user_feat_dict=self.train_dataset.user_feat_dict, item_feat_dict=self.train_dataset.item_feat_dict)
    
    def _get_dataset(self, flag='train'):
        interactions = pd.read_csv(self.cfg.dataset.interactions_path)
        user_features = pd.read_csv(self.cfg.dataset.user_features_path)
        item_features = pd.read_csv(self.cfg.dataset.item_features_path)
        return RecDataset(self.cfg, interactions, user_features, item_features, flag=flag)
    

    def vail(self, topk: int):
        flag = 'vali'
        self.model.eval()
        top1_list = []
        target_list = []
        if topk >= 1:
            topk_list = []

        item_vector = []
        log_vector = []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader)):
                (
                    user_id,
                    j,
                    user_feat,
                    id_seq,
                    feat_seq,
                    pos_seq,
                    pos_feat,
                    neg_seq,
                    neg_feat,
                    inter_time,
                    act_type,
                    token_type,
                ) = batch

                log_seq, target_item = self.model.vali(
                    user_id,
                    j,
                    user_feat,
                    id_seq,
                    feat_seq,
                    pos_seq,
                    pos_feat,
                    neg_seq,
                    neg_feat,
                    inter_time,
                    act_type,
                    token_type,
                )
                log_vector.append(log_seq)
                target_list.extend(target_item.cpu().numpy().tolist())
            
            for i, batch in tqdm(enumerate(self.item_dataset_loader), total=len(self.item_dataset_loader)):
                item_id, item_feat = batch
                item_emb = self.model.save_item_emb(item_id, item_feat)
                item_vector.append(item_emb)

            item_vector = torch.cat(item_vector, dim=0) # (item_num, dim)

            for i in range(len(log_vector)):
                if log_vector[i] is None:
                    continue
                log_vector[i] = log_vector[i]
                scores = torch.matmul(log_vector[i], item_vector.T)
                _, top1 = scores.topk(1, dim=-1)
                top1_list.extend(top1.squeeze(-1).cpu().numpy().tolist())




        y_pred = np.array(top1_list, dtype=np.int32)
        y_true = np.array(target_list, dtype=np.int32)

        precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }   

    def train(self):


        self.model.to(self.cfg.device)
        self.model.train()
        global_step = 0
        best_f1 = 0
        for epoch in range(self.cfg.train.epochs):
            epoch_loss = []
            for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                self.model.zero_grad()
                (           
                    user_id,
                    j,
                    user_feat,
                    id_seq,
                    feat_seq,
                    pos_seq,
                    pos_feat,
                    neg_seq,
                    neg_feat,
                    inter_time,
                    act_type,
                    token_type,
                ) = batch
                (
                    mainloss,
                    pos_score,
                    neg_score,
                    neg_var,
                    neg_max,
                ) = self.model(
                    user_id,
                    j,
                    user_feat,
                    id_seq,
                    feat_seq,
                    pos_seq,
                    pos_feat,
                    neg_seq,
                    neg_feat,
                    inter_time, 
                    act_type,
                    token_type,
                )
                mainloss.backward()
                self.optimizer.step()
                epoch_loss.append(mainloss.item())
                self.writer.add_scalar('Train/Loss', mainloss.item(), global_step)
                global_step += 1

            print(f"Epoch {epoch+1}/{self.cfg.train.epochs}, Loss: {sum(epoch_loss)/len(epoch_loss):.4f}")
            epoch_loss = []
            
            metrics = self.vail(topk=1)
            self.writer.add_scalar('Val/Precision', metrics['precision'], global_step)
            self.writer.add_scalar('Val/Recall', metrics['recall'], global_step)
            self.writer.add_scalar('Val/F1', metrics['f1'], global_step)
            print(f"Validation - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save(self.model.state_dict(), f'{self.cfg.save_path}/best_model.pth')
                print(f"Best model saved with F1: {best_f1:.4f}")
            


                
        
        
