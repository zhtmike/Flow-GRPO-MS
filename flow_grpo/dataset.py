import os
from typing import Optional

import numpy as np
from mindspore.dataset import Sampler


class TextPromptDataset:

    def __init__(self,
                 dataset: str,
                 split: str = 'train',
                 max_num: Optional[int] = None) -> None:
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        if max_num is not None:
            self.prompts = self.prompts[:max_num]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


class DistributedKRepeatSampler(Sampler):

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        # we should expect the sampler to be run infinitely, but somehow it does not have this behavior
        # we manuutally set num_samples to be big enough
        super().__init__(num_samples=len(dataset))
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = np.random.default_rng(self.seed + self.epoch)
            indices = g.permutation(len(self.dataset))[:self.m]
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = g.permutation(len(repeated_indices))
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            self.epoch += 1
            yield per_card_samples[self.rank]
