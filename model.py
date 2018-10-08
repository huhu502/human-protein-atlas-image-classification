from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss
from sklearn.metrics import f1_score
from torch import save
import numpy as np

from settings import neptune_context
import os


class ModelTrainer:
    def __init__(self, model,
                 optimizer,
                 criterion,
                 train_loader,
                 test_loader,
                 device,
                 epochs,
                 checkpoint_directory='/home/i008/'
                 ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trainer = create_supervised_trainer(model, optimizer, criterion, device)
        self.epochs = epochs
        self.checkpoint_directory = checkpoint_directory
        self.epoch_end_loss = None
        self.evaluator = create_supervised_evaluator(model, device=device,
                                                     metrics={'loss': Loss(criterion)})

        self.register_callbacks()

    def train(self):
        self.trainer.run(self.train_loader, max_epochs=self.epochs)

    def register_callbacks(self):
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self._callback_store_training_loss)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._callback_store_training_results)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._callback_checkpoint)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._epoch_end_callback_log)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self._batch_end_callback_log)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._batch_end_evaluate)

    def _callback_store_training_loss(self, engine):
        self.current_epoch = engine.state.epoch
        self.batch_end_loss = engine.state.output
        self.batch_number = engine.state.iteration

    def _callback_store_training_results(self, engine):
        if self.test_loader is not None:
            self.evaluator.run(self.test_loader)
            metrics = self.evaluator.state.metrics
            avg_nll = metrics['loss']
            self.epoch_end_loss = avg_nll

    def _callback_checkpoint(self, *args):
        model_name = str(self.model).split('(')[0]
        model_name += '_{}_{}'
        model_path = os.path.join(self.checkpoint_directory, model_name).format(self.current_epoch, self.epoch_end_loss)
        print("Storing model {}".format(model_path))
        save(self.model, model_path)

    def _epoch_end_callback_log(self, *args):
        neptune_context.channel_send('validation_loss_epoch_end', self.epoch_end_loss)
        neptune_context.channel_send("epoch", self.current_epoch)
        print("Finished epoch {} with loss {}".format(self.current_epoch, self.epoch_end_loss))

    def _batch_end_callback_log(self, *args):
        neptune_context.channel_send('loss', self.batch_end_loss)
        if self.batch_number % 100 == 0:
            print(self.batch_end_loss, self.batch_number)

    def _batch_end_evaluate(self, *args):
        TS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

        def evaluate_f1(model, test_loader, threshold=0.2):
            all_preds = []
            true = []
            model.eval()
            for b in test_loader:
                X, y = b
                X, y = X.cuda(), y.cuda()
                pred = model(X)
                all_preds.append(pred.sigmoid().cpu().data.numpy())
                true.append(y.cpu().data.numpy())

            P = np.concatenate(all_preds)
            R = np.concatenate(true)

            f_scores = []
            for t in TS:
                f1 = f1_score(P > t, R, average='macro')
                f_scores.append(f1)

            return f_scores

        self.f1_score = evaluate_f1(self.model, self.test_loader)

        for i, t in enumerate(TS):
            print(self.f1_score[i])
            neptune_context.channel_send("f1_score_validation_at{}".format(t), self.f1_score[i])
        self.model.train()
