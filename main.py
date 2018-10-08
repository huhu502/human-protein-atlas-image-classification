import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from neptune import Context
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

from architectures import get_resnet
from datasets import MultiBandMultiLabelDataset
from model import ModelTrainer

RANDOM_SEED = 666

LABEL_MAP = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"}

ctx = Context()

PATH_TO_IMAGES = '/media/i008/duzy/genom/train/'
PATH_TO_TEST_IMAGES = '/media/i008/duzy/genom/test/'
PATH_TO_META = '/media/i008/duzy/genom/train.csv'
SAMPLE_SUBMI = '/media/i008/duzy/genom/sample_submission.csv'

SEED = 666
DEV_MODE = False

df = pd.read_csv(PATH_TO_META)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
df_submission = pd.read_csv(SAMPLE_SUBMI)

if DEV_MODE:
    df_train, df_test = df_train[:1000], df_test[:100]

SIZE = 512
BS = 4

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

image_transform_train = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)

])

image_transform_test = transforms.Compose([
    transforms.Resize(SIZE),
    #             transforms.RandomVerticalFlip(),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomRotation(90)
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)

])
gtrain = MultiBandMultiLabelDataset(df_train, base_path=PATH_TO_IMAGES, image_transform=image_transform_train)
gtest = MultiBandMultiLabelDataset(df_test, base_path=PATH_TO_IMAGES, image_transform=image_transform_test)
gsub = MultiBandMultiLabelDataset(df_submission, base_path=PATH_TO_TEST_IMAGES, train_mode=False,
                                  image_transform=image_transform_test)

train_load = DataLoader(gtrain, collate_fn=gtrain.collate_func, batch_size=BS, num_workers=6)
test_load = DataLoader(gtest, collate_fn=gtest.collate_func, batch_size=BS, num_workers=6)
submission_load = DataLoader(gsub, collate_fn=gsub.collate_func, batch_size=BS, num_workers=6)

model = get_resnet()
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


criterion = nn.BCEWithLogitsLoss()
criterion = criterion.cuda()
# evaluator = create_supervised_evaluator(model,
#                                             device=device,
#                                             metrics={'loss': Loss(criterion)
#                                                     })
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
# trainer = create_supervised_trainer(model, optimizer, criterion, device=device)


neptune_context = Context()

if __name__ == '__main__':
    mlc = ModelTrainer(model, optimizer, criterion, train_load, test_load, device, 10)
    mlc.train()
