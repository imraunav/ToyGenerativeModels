import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from glob import glob
import numpy as np
import cv2


def remove_alpha_ch(x: torch.Tensor):
    assert x.shape[0] == 4
    alpha = x[3, :, :]
    x[:3, alpha < 0.8] = 255.0
    return x[:3, ...]


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.array):
        flip = np.random.choice([True, False], p=[self.p, 1.0 - self.p])
        if flip:
            img = img[:, ::-1]
        return img


class RandomCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        h, w, _ = img.shape
        assert h > self.height and w > self.width
        i = np.random.randint(0, h - self.height)
        j = np.random.randint(0, w - self.width)
        crop = img[i : i + self.height, j : j + self.width]
        assert crop.shape[:2] == (
            self.height,
            self.width,
        ), f"crop size:{crop.shape[:2]} smaller than set crop size{(self.height, self.width)}"
        return crop


class Resize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        img = cv2.resize(img, (self.height, self.width))
        return img


class ImageDataset(Dataset):
    def __init__(self, root: str, transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms

        pattern = os.path.join(root, "**/*.jpg")
        self.images = glob(pattern, recursive=True)
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        # change to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(img)

        return img


if __name__ == "__main__":
    path = "/Users/raunavghosh/Documents/code_projects/14-celebrity-faces-dataset/data"
    # path = "/Users/raunavghosh/Downloads/tinyface"
    ds_transforms = transforms.Compose(
        [
            RandomVerticalFlip(0.5),
            # RandomCrop(256, 512),
            Resize(256, 256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    ds = ImageDataset(path, ds_transforms)

    for i in range(len(ds)):
        img = ds[0]
        print(img.min(), img.max())
        print(img.dtype)
        # cv2.imshow("img", img[..., ::-1])
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()


# class LogoDataset(Dataset):
#     def __init__(self, root_dir, transforms=None):
#         super().__init__()
#         self.root_dir = root_dir
#         self.transforms = transforms

#         self.images = glob(os.path.join(root_dir, "*"))
#         self.len = len(self.images)

#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         image = Image.open(image)

#         if self.transforms is not None:
#             image = self.transforms(image)

#         return image

# if __name__ == "__main__":
#     path = "/Users/raunavghosh/Documents/code_projects/Logo_dataset/Logos"
#     ds_transforms = transforms.Compose(
#         [
#             transforms.Resize(512),
#             transforms.ToTensor(),
#             transforms.Lambda(
#                 lambda x: (
#                     torch.cat([x, x, x], dim=0)
#                     if (x.shape[0] == 1 or len(x.shape) < 3)
#                     else x
#                 )
#             ),
#             transforms.Lambda(lambda x: remove_alpha_ch(x) if x.shape[0] > 3 else x),
#             transforms.Normalize((0.5,), (0.5,)),
#         ]
#     )
#     ds = LogoDataset(path, ds_transforms)
#     from einops import rearrange

#     for sample in ds:
#         # print(sample.shape)
#         # cv2.imshow("Logo", rearrange(sample.numpy(), "c h w -> h w c")[:, :, ::-1])
#         # cv2.waitKey(1000)
#         # cv2.destroyAllWindows()
#         pass
