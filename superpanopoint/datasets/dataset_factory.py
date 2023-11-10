from typing import List, Literal

from torch.utils.data import random_split

from superpanopoint import Settings
from superpanopoint.datasets import BaseDataset, DataSample
from superpanopoint.datasets.synthetic import SyntheticDataset
from superpanopoint.utils.logger import setup_logger

logger = setup_logger(__name__)

Mode = Literal["tran", "val", "test"]


class DatasetFactory:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model_name = cfg["model"]["name"]
        train_subset, val_subset, test_subset = self._split_data_sources(cfg["dataset"]["sources"])
        self.mode_name_to_subset = {
            "train": train_subset,
            "val": val_subset,
            "test": test_subset,
        }

    def create_dataset(self, mode: Mode) -> BaseDataset:
        dataset_subset = self.mode_name_to_subset[mode]
        if self.model_name == "synthetic":
            return SyntheticDataset(dataset_subset, **self.cfg["dataset"][mode])
        else:
            raise NotImplementedError

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
        imgs_folder = Settings.data_dir / folder_name / Settings.img_dir_name
        points_folder = Settings.data_dir / folder_name / Settings.points_dir_name
        img_files = list(imgs_folder.glob("*"))
        valid_samples = []
        for img_file in img_files:
            corresponding_points_file = points_folder / (img_file.stem + ".txt")
            if not corresponding_points_file.exists():
                logger.warn(f"File {img_file.name} does not have corresponding corner points file")
            valid_samples.append(DataSample(img=img_file, points=corresponding_points_file))
        return valid_samples
