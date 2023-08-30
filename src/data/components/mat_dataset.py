import json
import os
from typing import Any, List, Tuple

# import sys
# sys.path.append("/media/d5/7D1922F98D178B12/hz/Code/img2pbr")
import hydra
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MatMatchingDataset(Dataset):
    """A material matching(aka.

    classification) dataset.
    """

    def __init__(
        self,
        data_root_dir: str,
        data_list_file_dir: str,
        subdataset_names: List[str],
        class_def_json_path: str,
        list_file_prefix: str,
        mode: str = "train",
        img_transforms: List = [],
        debug_on: bool = False,
        debug_data_num: int = 100,
    ) -> None:
        """Initialize a `MatMatchingDataset`.

        :param data_root_dir: The root directory of data.
        :param data_list_file_dir: The directory of data list file, used to select samples.
        :param subdataset_names: The subdatasets to be used in [SU_mat, D5_mat...].
        :param class_def_json_path: The json file path of class definition.
        :param list_file_prefix: The prefix of material data list file.
        :param mode: The training mode of this dataset used for, train or test.
        :param img_transforms: The img transformations for data augmentation.
        :param debug_on: Build the dataset for debugging or testing.
        :param debug_data_num: The number of samples for debugging or testing. Only effective when `debug_on=True`.
        """
        super().__init__()
        self.data_root_dir = data_root_dir
        self.data_list_file_dir = data_list_file_dir
        assert len(subdataset_names) > 0, "No sub dataset found!!!"
        self.subdataset_names = subdataset_names
        assert os.path.isfile(class_def_json_path), "The class definition file path is not exist."
        self.class_def_path = class_def_json_path
        self.list_file_prefix = list_file_prefix
        self.mode = mode
        self.debug_on = debug_on
        self.debug_data_num = debug_data_num if debug_on else 1000000000
        self._tform = self._set_transforms(img_transforms)
        self._data = self._preprocess()

    def __len__(self) -> int:
        """Get the number of samples in dataset.

        :return: The number of samples in dataset.
        """
        return len(self._data)

    def _preprocess(self) -> List[List[Any]]:
        """Process the raw data.

        :return: The list of processed samples.
        """
        # list to be returned
        data = []

        # load data of different subdatasets with one data_list_file

        data_list_file_path = os.path.join(
            self.data_list_file_dir, self.list_file_prefix + "_" + self.mode + ".txt"
        )
        with open(self.class_def_path) as f:
            class2idx = json.load(f)
        cnt = 0
        for dataset_name in self.subdataset_names:
            # sample name list
            sample_names = []
            with open(data_list_file_path) as f:
                lines = f.readlines()
                for line in lines:
                    class_slash_name = line.strip()
                    sample_names.append(class_slash_name)
            assert len(sample_names) > 0, "No data here"

            img_type = ("jpg", "png")
            # load all material images to be classified
            for class_slash_name in sample_names:
                # debug with a subset of whole dataset
                if len(data) >= self.debug_data_num:
                    break

                label, name = class_slash_name.split("/")

                img_path = os.path.join(self.data_root_dir, dataset_name, name + ".{}")
                for suffix in img_type:
                    if os.path.isfile(img_path.format(suffix)):
                        img_path = img_path.format(suffix)
                        break

                if not os.path.isfile(img_path):
                    continue

                # load input img and it's class label
                input = Image.open(img_path)  # PIL.Image，h w 3
                input = input.convert("RGB")
                input = input.resize((512, 512))
                gt = int(class2idx[label])

                data.append([name, input, gt])
                cnt += 1

        return data

    def _set_transforms(self, img_transforms: list = [transforms, ...]) -> transforms:
        """Set the transforms, such as resize(), ToTensor(), for imgs.

        :return: The composed img transforms in torchvision.transforms.
        """
        img_transforms = [hydra.utils.instantiate(tt) for tt in img_transforms]  # still have bug
        img_transforms.extend(
            [
                transforms.ToTensor(),  # row_data -> (0, 1)
                transforms.Lambda(lambda x: x * 2.0 - 1.0),  # (0, 1) -> (-1, 1)
            ]
        )
        return transforms.Compose(img_transforms)

    def _img_process(self, image: Image) -> Image:
        """Apply the transforms to imgs before entering the model.

        :return: The transformed imgs.
        """
        return self._tform(image)

    def __getitem__(self, index) -> Tuple[Any]:
        name, input, gt = self._data[index]
        input = self._img_process(input)
        gt = torch.tensor(gt, dtype=torch.long)
        return (name, input, gt)


if __name__ == "__main__":
    dataset = MatMatchingDataset(
        data_root_dir="/media/d5/7D1922F98D178B12/hz/DataSet/ai_mat",
        data_list_file_dir="/media/d5/7D1922F98D178B12/hz/DataSet/ai_mat",
        subdataset_names=["SU_mat", "D5_mat"],
        class_def_json_path="/media/d5/7D1922F98D178B12/hz/DataSet/ai_mat/mat_class.json",
        list_file_prefix="classify",
        mode="train",
        debug_on=False,
    )
    print(len(dataset))
