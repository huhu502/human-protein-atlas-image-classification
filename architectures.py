from ignite.engine import Events
from ignite.engine import create_supervised_trainer
from torch import nn
from torchvision.models import resnet


def get_resnet(n_classes=28, resnet_version='resnet152', image_channels=3, pretrained=True):
    """
    :param n_classes:
    :param resnet_version in ['resnet152', 'resnet50', 'resnet18', 'resnet101']
    :param image_channels:
    :param pretrained:
    :return:
    """
    assert resnet_version in ['resnet' + str(i) for i in [18, 50, 101, 152]]
    model = getattr(resnet, resnet_version)(pretrained=pretrained)
    for p in model.parameters():
        p.requires_grad = True
    inft = model.fc.in_features
    model.fc = nn.Linear(in_features=inft, out_features=n_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if image_channels is not 3:
        model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)

    return model


class MultiLabelClassifyModel:
    def __init__(self, model,
                 optimizer,
                 criterion,
                 train_loader,
                 test_loader,
                 device,
                 epochs
                 ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trainer = create_supervised_trainer(model, optimizer, criterion, device)
        self.epochs = epochs
        self.register_callbacks()

    def train(self):
        self.trainer.run(self.train_loader, max_epochs=self.epochs)

    def register_callbacks(self):
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.log_training_loss)

    def log_training_loss(self, engine):
        iter = (engine.state.iteration - 1) % len(self.train_loader) + 1
        #         ctx.channel_send('loss', engine.state.output)
        if iter % 100 == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(self.train_loader), engine.state.output))


