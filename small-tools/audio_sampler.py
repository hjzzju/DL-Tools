class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    def __init__(self, lengths, batch_size=15000):
        self.lengths = lengths
        _, self.sorted_indices = torch.sort(torch.LongTensor(lengths))
        self.batch_size = batch_size
        self.fixed_batches = []
        self.indices_group_by_lengths = [[] for _ in range(21)]
        self.length_boundary = [x * 50 for x in range(1, 21)] + [100000]  # [50, ... 1000, 100000]

        now_bin_id = 0
        for i in self.sorted_indices:
            while lengths[i] > self.length_boundary[now_bin_id]:
                now_bin_id += 1
            self.indices_group_by_lengths[now_bin_id].append(i.item())

    def __iter__(self):
        all_indices = []
        for i in range(len(self.indices_group_by_lengths)):
            random.shuffle(self.indices_group_by_lengths[i])
            all_indices += self.indices_group_by_lengths[i]

        batches = []
        total_lengths = 0
        new_batches = []
        for i in all_indices:
            if self.lengths[i] + total_lengths > self.batch_size:
                new_batches.append(batches)
                batches = []
                total_lengths = 0
            batches.append(i)
            total_lengths += self.lengths[i]

        if len(batches) >= 1:
            new_batches.append(batches)

        random.shuffle(new_batches)
        return iter(new_batches)

    def __len__(self):
        return len(self.sorted_indices)


train_loader = DataLoader(trainset, num_workers=24,
                          shuffle=False,
                          sampler=train_sampler(),
                          batch_size=1,
                          pin_memory=False,
                          drop_last=False,
                          collate_fn=collate_fn)