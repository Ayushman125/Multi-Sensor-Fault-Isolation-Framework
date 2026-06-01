from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class KITTITemporalVelocityDataset(Dataset):
    """Temporal KITTI loader that pairs adjacent frames from the flattened object split.

    The workspace exposes image_2 and label_2 folders without raw drive sequence IDs, so
    consecutive filenames are treated as the most reproducible temporal ordering available
    in this repository.
    """

    def __init__(self, root_dir='./data/kitti', split='training', labeled=True, delta_t=0.1):
        self.root_dir = Path(root_dir)
        self.split = split
        self.labeled = labeled
        self.delta_t = delta_t
        self.image_dir = self.root_dir / split / 'image_2'
        self.label_dir = self.root_dir / split / 'label_2'
        self.transform = transforms.Compose([
            transforms.Resize((64, 192)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        image_paths = sorted(
            list(self.image_dir.glob('*.png'))
            + list(self.image_dir.glob('*.jpg'))
            + list(self.image_dir.glob('*.jpeg'))
        )
        if len(image_paths) < 2:
            raise FileNotFoundError(f'Need at least two KITTI images in {self.image_dir}')

        self.samples = []
        for index in range(len(image_paths) - 1):
            image_t = image_paths[index]
            image_t1 = image_paths[index + 1]
            label_t = self.label_dir / f'{image_t.stem}.txt'
            label_t1 = self.label_dir / f'{image_t1.stem}.txt'
            if self.labeled:
                if label_t.exists() and label_t1.exists():
                    self.samples.append((image_t, image_t1, label_t, label_t1))
            else:
                self.samples.append((image_t, image_t1, None, None))

        if not self.samples:
            raise FileNotFoundError(f'No usable KITTI temporal samples found in {self.image_dir}')

    def __len__(self):
        return len(self.samples)

    def _read_primary_depth(self, label_path):
        if label_path is None or not label_path.exists():
            return None

        best_area = -1.0
        best_depth = None
        with open(label_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                class_name = parts[0]
                if class_name == 'DontCare':
                    continue
                left, top, right, bottom = map(float, parts[4:8])
                area = max(0.0, (right - left) * (bottom - top))
                if area > best_area:
                    best_area = area
                    best_depth = abs(float(parts[13]))
        return best_depth

    def __getitem__(self, idx):
        image_t_path, image_t1_path, label_t_path, label_t1_path = self.samples[idx]
        img_t = self.transform(Image.open(image_t_path).convert('RGB'))
        img_t1 = self.transform(Image.open(image_t1_path).convert('RGB'))
        temporal_image = torch.cat([img_t, img_t1], dim=0)

        z_t = self._read_primary_depth(label_t_path)
        z_t1 = self._read_primary_depth(label_t1_path)
        if z_t is not None and z_t1 is not None:
            calculated_speed = abs((z_t1 - z_t) / self.delta_t)
            target_bucket = int(np.clip(calculated_speed // 2.5, 0, 9))
        else:
            calculated_speed = 0.0
            target_bucket = -1

        v_odo_raw = calculated_speed + np.random.normal(loc=0.0, scale=0.2)
        v_speed_raw = calculated_speed + np.random.normal(loc=0.0, scale=0.2)

        p_odo_vector = np.zeros(10, dtype=np.float32)
        p_speed_vector = np.zeros(10, dtype=np.float32)
        p_odo_vector[int(np.clip(v_odo_raw // 2.5, 0, 9))] = 1.0
        p_speed_vector[int(np.clip(v_speed_raw // 2.5, 0, 9))] = 1.0

        return (
            temporal_image,
            torch.tensor(p_odo_vector, dtype=torch.float32),
            torch.tensor(p_speed_vector, dtype=torch.float32),
            torch.tensor(target_bucket, dtype=torch.long),
        )
