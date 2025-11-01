from torch.utils.data import Sampler
import random

class BucketSampler(Sampler):
    def __init__(self, dataset, bs_len=128*128, bucket_size=256, shuffle=True):
        self.dataset = dataset
        self.bs_len = bs_len
        self.bucket_size = bucket_size
        self.shuffle = shuffle


        self.user_id_to_idx = {
            uid: idx for idx, uid in enumerate(dataset.interactions_user_ids)
        }

        user_lens = [(uid, dataset.user_seq_lens[uid]) for uid in dataset.interactions_user_ids]
        user_lens.sort(key=lambda x: x[1], reverse=False)  

        self.sorted_user_ids = [u for u, _ in user_lens]

        self.buckets = [
            self.sorted_user_ids[i:i + bucket_size]
            for i in range(0, len(self.sorted_user_ids), bucket_size)
        ]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.buckets)

        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)


            bucket_lens = [self.dataset.user_seq_lens[u] for u in bucket]
            max_len = max(bucket_lens)
            batch_size = max(1, self.bs_len // max_len)

            for i in range(0, len(bucket), batch_size):
                batch_indices = [self.user_id_to_idx[u] for u in bucket[i:i+batch_size]]
                yield batch_indices

    def __len__(self):
        total_batches = 0
        for bucket in self.buckets:
            bucket_lens = [self.dataset.user_seq_lens[u] for u in bucket]
            max_len = max(bucket_lens)
            batch_size = max(1, self.bs_len // max_len)
            total_batches += (len(bucket) + batch_size - 1) // batch_size
        return total_batches