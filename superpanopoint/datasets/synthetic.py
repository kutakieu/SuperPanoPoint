from typing import List

from . import BaseDataset, DataSample


class PointsDataset(BaseDataset):
    def __init__(self, data_samples: List[DataSample], **kwargs):
        super().__init__(**kwargs)
        self.data_samples = data_samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index: int):
        sample = self.data_samples[index]
        return sample.load_img(), sample.load_points()

    def __repr__(self):
        return f"PointDataset: {len(self)} samples"
