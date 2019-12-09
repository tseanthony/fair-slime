import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

class SSL_IMG_JIGSAW(object):
    def __init__(self, indices, perms_path):
        self.indices = set(indices)
        self.perms = np.load(perms_path)

    def _transform(self, img, count=3, size=255):
        img = T.Compose([
            T.CenterCrop(85),
            T.Resize(size),
        ])(img)

        tiles = []
        cs = size // count       # crop size
        for i in range(count):
            for j in range(count):
                tile = TF.crop(img, cs*i, cs*j, cs, cs)
                tile = self._tile_transform(tile)
                tiles.append(tile)

        label = np.random.randint(0, self.perms.shape[0])
        tiles = [tiles[i] for i in self.perms[label]]
        return torch.stack(tiles), label

    def _color_channels_jitter(self, img):
        h, w = img.size
        channels = img.split()
        cropper = T.RandomCrop((h - 2, w - 2))
        return Image.merge('RGB', [cropper(c) for c in channels])

    def _tile_transform(self, tile):
        tile = T.Compose([
            T.Lambda(self._color_channels_jitter),
            T.RandomCrop(64),
            T.ToTensor(),
        ])(tile)
        return TF.normalize(tile, mean=tile.mean((1, 2)), std=tile.std((1, 2)))

    def __call__(self, sample):
        data, labels = [], []
        indices = self.indices if self.indices else set(range(len(sample["data"])))
        for idx in range(len(sample["data"])):
            if idx in indices:
                img, label = self._transform(sample["data"][idx])
                data.append(img)
                labels.append(label)

        sample["data"] = data
        sample["label"] = labels

        return sample
