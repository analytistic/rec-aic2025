from torch.utils.data import Sampler

import random



class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, bucket_size=100, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        user_lens = [(uid, dataset.user_seq_lens[uid]) for uid in dataset.interactions_user_ids]

        user_lens.sort(key=lambda x: x[1])

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

            for i in range(0, len(bucket), self.batch_size):
                yield [self.dataset.interactions_user_ids.index(u) for u in bucket[i:i+self.batch_size]]


    def __len__(self):
        return sum(len(bucket)//self.batch_size for bucket in self.buckets)


