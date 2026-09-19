import os
import sys
import json
import random
import time

import albumentations as A
import numpy as np
from PIL import ImageFile
from skimage import io
from torch.utils.data import Dataset
from tqdm import tqdm

from transforms import Preprocessing

ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_CELL_LABEL_KEYWORDS = ("\u5370\u6212\u7ec6\u80de",)


def _resolve_dataset_root(data_root, dataset):
    if os.path.isabs(dataset):
        return dataset
    return os.path.join(data_root, dataset)


def _find_json_path(point_dir, image_file):
    base_name = os.path.splitext(image_file)[0]
    candidates = (
        f"{base_name}.jpg.json",
        f"{image_file}.json",
        f"{base_name}.json",
    )
    for name in candidates:
        path = os.path.join(point_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(point_dir, candidates[0])


def _count_cell_points(json_path, label_keywords):
    with open(json_path, encoding="utf-8") as f:
        annotations = json.loads(f.read())

    count = 0
    for ann in annotations.get("annotation", []):
        label_values = ann.get("label", [])
        label = str(label_values[0]) if label_values else ""
        if any(keyword in label for keyword in label_keywords):
            count += 1
    return count


def img_loader(data_root, dataset, num_classes, phase):
    dataset_root = _resolve_dataset_root(data_root, dataset)
    img_dir = os.path.join(dataset_root, f"{phase}_image")
    point_dir = os.path.join(dataset_root, f"{phase}_point")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"image directory not found: {img_dir}")
    if not os.path.isdir(point_dir):
        raise FileNotFoundError(f"point directory not found: {point_dir}")

    data = []
    files = []
    reader = tqdm(sorted(os.listdir(img_dir)), file=sys.stdout)
    time_string = time.strftime("[%D-%H:%M:%S]", time.localtime())
    reader.set_description(f"{time_string} loading {phase} data")

    for file in reader:
        image_path = os.path.join(img_dir, file)
        if not os.path.isfile(image_path):
            continue

        json_path = _find_json_path(point_dir, file)
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found - {json_path}")
            continue

        files.append(image_path)
        data.append(json_path)

    return data, files


class DataFolder(Dataset):
    def __init__(self, data_root, dataset, num_classes, phase, data_transform,
                 max_samples=0, mini_seed=1229, label_keywords=None, skip_empty=False):
        self.data_root = data_root
        self.dataset = dataset
        self.num_classes = num_classes
        self.phase = phase
        self.dataset_root = _resolve_dataset_root(data_root, dataset)
        self.label_keywords = tuple(label_keywords or DEFAULT_CELL_LABEL_KEYWORDS)
        self.data, self.files = img_loader(data_root, dataset, num_classes, phase)
        self.original_size = len(self.data)
        self.skipped_empty = 0

        if skip_empty:
            kept_data = []
            kept_files = []
            for json_path, image_path in zip(self.data, self.files):
                if _count_cell_points(json_path, self.label_keywords) > 0:
                    kept_data.append(json_path)
                    kept_files.append(image_path)
            self.skipped_empty = len(self.data) - len(kept_data)
            self.data = kept_data
            self.files = kept_files
            print(
                f"[SkipEmptyDataset] {phase}: used {len(self.data)} / {self.original_size} samples, "
                f"skipped_empty={self.skipped_empty}, root={self.dataset_root}"
            )

        self.size_after_empty_filter = len(self.data)

        if max_samples and max_samples > 0:
            indices = list(range(len(self.data)))
            random.Random(mini_seed).shuffle(indices)
            indices = indices[:max_samples]
            self.data = [self.data[i] for i in indices]
            self.files = [self.files[i] for i in indices]
            print(f"[MiniDataset] {phase}: selected {len(self.data)} / {self.original_size} samples from {self.dataset_root}")

        if not self.data:
            raise RuntimeError(f"no valid samples found in {self.dataset_root} for phase={phase}")

        self.data_transform = data_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        index = index % len(self.data)
        try:
            sample = self.read_data(self.data[index], self.files[index])
            sample = self.data_transform(sample)
        except Exception as exc:
            print(f"Error loading data index={index}, image={self.files[index]}: {exc}")
            sample = self.__getitem__(index + 1)

        if self.phase == "train" and (sample[0].shape[1] != 1080 or sample[0].shape[2] != 1920):
            print(self.files[index], " crop size error!")
            sample = self.__getitem__(index + 1)

        return sample

    def read_data(self, data, files):
        keys = ["image", "keypoints"] + [f"keypoints{i}" for i in range(1, self.num_classes)]
        values = [io.imread(files)]

        with open(data, encoding="utf-8") as f:
            annotations = json.loads(f.read())

        points = []
        for ann in annotations.get("annotation", []):
            label_values = ann.get("label", [])
            label = str(label_values[0]) if label_values else ""
            if any(keyword in label for keyword in self.label_keywords):
                x = float(ann["position"]["x"][0])
                y = float(ann["position"]["y"][0])
                points.append([x, y])

        values.append(np.array(points).reshape(-1, 2) if points else np.empty((0, 2)))
        return dict(zip(keys, values))


def _select_dataset(args, image_set):
    train_dataset = getattr(args, "train_dataset", "") or args.dataset
    test_dataset = getattr(args, "test_dataset", "") or args.dataset
    eval_dataset = getattr(args, "eval_dataset", "") or train_dataset
    final_test_dataset = getattr(args, "final_test_dataset", "") or test_dataset
    if image_set == "train":
        return train_dataset
    if image_set == "eval":
        return eval_dataset
    if image_set == "final_test":
        return final_test_dataset
    return test_dataset


def build_dataset(args, image_set):
    data_root = getattr(args, "data_root", "./datasets")
    dataset = _select_dataset(args, image_set)
    dataset_root = _resolve_dataset_root(data_root, dataset)
    mean_std_path = getattr(args, "mean_std_path", "") or os.path.join(dataset_root, "mean_std.npy")
    if not os.path.exists(mean_std_path):
        raise FileNotFoundError(f"mean/std file not found: {mean_std_path}")

    mean, std = np.load(mean_std_path)
    additional_targets = {}
    for i in range(1, args.num_classes):
        additional_targets.update({f"keypoints{i}": "keypoints"})

    actual_phase = "train" if image_set == "train" else "test"

    if actual_phase == "train":
        augmentor = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=0, shift_limit=0,
                               border_mode=0, value=0, p=0.5),
            A.RandomCrop(height=1080, width=1920, always_apply=True),
        ], p=1, keypoint_params=A.KeypointParams(format="xy"),
            additional_targets=additional_targets)
        transform = Preprocessing(mean, std, augmentor)
    elif actual_phase == "test":
        transform = Preprocessing(mean, std)
    else:
        raise NotImplementedError

    max_samples = getattr(args, "mini_train", 0) if actual_phase == "train" else getattr(args, "mini_test", 0)
    label_keywords = getattr(args, "cell_label_keywords", None)
    skip_empty = False
    if actual_phase == "train":
        skip_empty = getattr(args, "skip_empty_train", False)
    elif image_set in ("eval", "test", "final_test"):
        skip_empty = getattr(args, "skip_empty_eval_dataset", False)
    data_folder = DataFolder(
        data_root=data_root,
        dataset=dataset,
        num_classes=args.num_classes,
        phase=actual_phase,
        data_transform=transform,
        max_samples=max_samples,
        mini_seed=getattr(args, "mini_seed", 1229),
        label_keywords=label_keywords,
        skip_empty=skip_empty,
    )
    data_folder.mean_std_path = mean_std_path
    return data_folder
