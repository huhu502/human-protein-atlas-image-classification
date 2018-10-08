import pathlib
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from settings import LABEL_MAP
import PIL


class MultiBandMultiLabelDataset(Dataset):
    BANDS_NAMES = ['_red.png', '_green.png', '_blue.png', '_yellow.png']

    def __len__(self):
        return len(self.images_df)

    def __init__(self, images_df,
                 base_path,
                 image_transform,
                 augmentator=None,
                 train_mode=True,
                 n_channels=4
                 ):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)

        self.images_df = images_df.copy()
        self.image_transform = image_transform
        self.augmentator = augmentator
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.mlb = MultiLabelBinarizer(classes=list(LABEL_MAP.keys()))
        self.train_mode = train_mode
        self.n_channels = n_channels

    def __getitem__(self, index):
        y = None
        X = self._load_multiband_image(index)
        if self.train_mode:
            y = self._load_multilabel_target(index)

        # augmentator can be for instance imgaug augmentation object
        if self.augmentator is not None:
            X = self.augmentator(X)

        X = self.image_transform(X)

        return X, y

    def _load_multiband_image(self, index):
        row = self.images_df.iloc[index]
        image_bands = []
        for band_name in self.BANDS_NAMES:
            p = str(row.Id.absolute()) + band_name
            pil_channel = PIL.Image.open(p)
            image_bands.append(pil_channel)

        # lets pretend its a RBGA image to support 4 channels
        band4image = PIL.Image.merge('RGBA', bands=image_bands[:])
        return band4image

    def _load_multilabel_target(self, index):
        return list(map(int, self.images_df.iloc[index].Target.split(' ')))

    def collate_func(self, batch):
        labels = None
        images = [x[0] for x in batch]

        if self.train_mode:
            labels = [x[1] for x in batch]
            labels_one_hot = self.mlb.fit_transform(labels)
            labels = torch.FloatTensor(labels_one_hot)

        image_stack = torch.stack(images)
        image_stack[:, 0, :, :] + image_stack[:, 3, :, :] / 2
        image_stack[:, 1, :, :] + image_stack[:, 3, :, :] / 2
        image_stack = image_stack[:, :3, :, :]

        return image_stack, labels
