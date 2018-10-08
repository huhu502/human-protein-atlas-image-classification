from torchvision import transforms

image_transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor()

])

image_transform_test = transforms.Compose([
    transforms.Resize(256),
    #             transforms.RandomVerticalFlip(),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomRotation(90)
    transforms.ToTensor()

])
