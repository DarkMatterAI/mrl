# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/08_agent.ipynb (unless otherwise specified).

__all__ = ['Agent']

# Cell
from .imports import *
from .core import *
from .torch_imports import *
from .torch_core import *

# Cell

class Agent():
    def __init__(self, model, vocab, loss_function, dataset):
        self.model = model
        self.base_model = copy.deepcopy(model)
        to_device(self.model)
        to_device(self.base_model)

        self.vocab = vocab
        self.dataset = dataset
        self.opt = self.get_opt()
        self.loss_function = loss_function

    def get_opt(self, **optim_kwargs):
        return optim.Adam(self.model.parameters(), **optim_kwargs)

    def train_supervised(self, bs, epochs, lr, percent_valid=0.05):

        train_ds, valid_ds = self.dataset.split(percent_valid)

        train_dl = train_ds.dataloader(bs)
        valid_dl = valid_ds.dataloader(bs)

        scheduler = optim.lr_scheduler.OneCycleLR(self.opt, max_lr=lr,
                                                 steps_per_epoch=len(train_dl), epochs=10)

        print('Epoch\tTrain\tValid')
        for epoch in range(epochs):
            train_losses = []
            for i, batch in enumerate(train_dl):
                x,y = batch
                if not type(x)==list:
                    x = [x]

                output = self.model(*x)
                self.opt.zero_grad()
                loss = self.loss_function(output, y)
                loss.backward()
                opt.step()
                scheduler.step()
                train_losses.append(loss.detach().cpu())

            with torch.no_grad():
                valid_losses = []
                for i, batch in enumerate(valid_dl):
                    x,y = batch
                    if not type(x)==list:
                        x = [x]

                    output = self.model(*x)
                    loss = self.loss_function(output, y)
                    valid_losses.append(loss.detach().cpu())

            train_loss = smooth_batches(train_losses)
            valid_loss = smooth_batches(valid_losses)

            print(f'{epoch}\t{train_loss:.2f}\t{valid_loss:.2f}')


    def update_dataset(self, dataset):
        self.dataset = dataset

    def update_dataset_from_inputs(self, *dataset_inputs):
        dataset = self.dataset.new(*dataset_inputs)
        self.update_dataset(dataset)

    def reconstruct(self, preds):
        return maybe_parallel(self.vocab.reconstruct, preds)

