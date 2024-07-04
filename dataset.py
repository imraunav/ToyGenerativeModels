import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from torchvision.transforms import (
    Compose,
    Resize,
    ToPILImage,
    ToTensor,
    Lambda,
    Normalize,
)
from PIL import Image


class SimpleDataset(Dataset):
    """
    A simple dataset class to load images from a directorcy without class labels
    """

    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms

        self.files = glob(os.path.join(path, "*.jpg"))
        self.len = len(self.files)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fpath = self.files[index]
        img = Image.open(fpath)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0  # 0 class to be consistent with x, y return format


def get_dataset():
    """
    Return a full dataset instance of CelebA with images of size (64, 64)
    """
    data_transforms = Compose(
        [
            Resize((64, 64)),
            ToTensor(),
            Normalize(
                (0.5,), (0.5,)
            ),  # normalize to range of [-1, 1] standard image generation range
        ]
    )

    ds = SimpleDataset(
        "./celeba50k",
        data_transforms,
    )
    return ds


if __name__ == "__main__":
    # test code
    ds = get_dataset()
    # print(len(ds)) # 50000
    # x = ds[0]
    # print(x.shape) # (3, 64, 64)

    # find dataset mean and std for all channels
    loader = DataLoader(ds, batch_size=50)
    mean = torch.zeros((3,))
    std = torch.zeros((3,))
    count = 0
    for x in loader:
        count += 1
        mean += x.mean(dim=(0, 2, 3))
        std += x.std(dim=(0, 2, 3))

    print(f"Mean: {mean / count}")  # Mean: tensor([0.5168, 0.4154, 0.3625])
    print(f"Std: {std / count}")  # Std: tensor([0.2962, 0.2673, 0.2615])
