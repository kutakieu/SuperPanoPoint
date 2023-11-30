from typing import List, Literal

from PIL import Image
from torch.utils.data import random_split

from superpanopoint import Settings
from superpanopoint.datasets import BaseDataset, DataSample
from superpanopoint.datasets.homographic import HomographicDataset
from superpanopoint.datasets.synthetic import SyntheticDataset
from superpanopoint.models.detector import Predictor
from superpanopoint.utils.logger import get_logger

logger = get_logger(__name__)

Mode = Literal["tran", "val", "test"]


class DatasetFactory:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        train_subset, val_subset, test_subset = self._split_data_sources(cfg["dataset"]["sources"])
        self.mode_name_to_subset = {
            "train": train_subset,
            "val": val_subset,
            "test": test_subset,
        }

    def create_dataset(self, mode: Mode) -> BaseDataset:
        dataset_subset = self.mode_name_to_subset[mode]
        if self.cfg.dataset.type == "synthetic":
            return SyntheticDataset(dataset_subset, **self.cfg["dataset"][mode], **self.cfg["dataset"][mode]["aug"])
        elif self.cfg.dataset.type == "homographic":
            detector = Predictor(self.cfg, self.cfg.dataset.detector_model, device='cpu')
            return HomographicDataset(dataset_subset, point_detector=detector, **self.cfg["dataset"][mode], **self.cfg["dataset"][mode]["aug"])

    def _split_data_sources(self, cfg_data_sources):
        train_subset = val_subset = test_subset = None
        for data_source_type in cfg_data_sources.keys():
            valid_sampels = self._extract_valid_samples(cfg_data_sources[data_source_type]["folder_name"])
            cur_train_subset, cur_val_subset, cur_test_subset = random_split(
                valid_sampels,
                [
                    cfg_data_sources[data_source_type]["train"],
                    cfg_data_sources[data_source_type]["val"],
                    cfg_data_sources[data_source_type]["test"],
                ],
            )
            train_subset = cur_train_subset if train_subset is None else (train_subset + cur_train_subset)
            val_subset = cur_val_subset if val_subset is None else (val_subset + cur_val_subset)
            test_subset = cur_test_subset if test_subset is None else (test_subset + cur_test_subset)
        return train_subset, val_subset, test_subset

    def _extract_valid_samples(self, folder_name: str) -> List[DataSample]:
        imgs_folder = Settings().data_dir / folder_name / Settings().img_dir_name
        points_folder = Settings().data_dir / folder_name / Settings().points_dir_name
        img_files = list(imgs_folder.glob("*"))
        valid_samples = []
        for img_file in img_files:
            if min(Image.open(img_file).size) < self.cfg.dataset.get("min_img_size", 360):
                continue
            points_file = points_folder / f"{img_file.stem}.json" if (points_folder / f"{img_file.stem}.json").exists() else None
            valid_samples.append(DataSample(img_file=img_file, points_file=points_file))
        print('num samples:', len(valid_samples))
        return valid_samples
