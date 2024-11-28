
import torch
import numpy as np

import stork.nodes.base
from stork.models.base import RecurrentSpikingModel
from stork import generators
from stork import loss_stacks


class DoubleInputRecSpikingModel(RecurrentSpikingModel):
    def __init__(
        self,
        batch_size,
        nb_time_steps,
        nb_inputs,
        nb_outputs,
        device=torch.device("cpu"),
        dtype=torch.float,
        sparse_input=False,
    ):
        super(RecurrentSpikingModel, self).__init__()
        self.batch_size = batch_size
        self.nb_time_steps = nb_time_steps
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.device = device
        self.dtype = dtype

        self.fit_runs = []

        self.groups = []
        self.connections = []
        self.devices = []
        self.monitors = []
        self.hist = []

        self.optimizer = None
        self.input_group = None
        self.output_group = None
        self.sparse_input = sparse_input

    def configure(
        self,
        input,
        output,
        loss_stack=None,
        optimizer=None,
        optimizer_kwargs=None,
        scheduler=None,
        scheduler_kwargs=None,
        generator1=None,
        generator2=None,
        time_step=1e-3,
        wandb=None,
    ):
        self.input_group = input
        self.output_group = output
        self.time_step = time_step
        self.wandb = wandb

        if loss_stack is not None:
            self.loss_stack = loss_stack
        else:
            self.loss_stack = loss_stacks.SumOfSoftmaxCrossEntropy()

        if generator1 is None:
            self.data_generator1_ = generators.StandardGenerator()
        else:
            self.data_generator1_ = generator1

        if generator2 is None:
            self.data_generator2_ = generators.StandardGenerator()
        else:
            self.data_generator2_ = generator2

        # configure data generator
        self.data_generator1_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )
        self.data_generator2_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )

        for o in self.groups + self.connections:
            o.configure(
                self.batch_size,
                self.nb_time_steps,
                self.time_step,
                self.device,
                self.dtype,
            )

        if optimizer is None:
            optimizer = torch.optim.Adam

        if optimizer_kwargs is None:
            optimizer_kwargs = dict(lr=1e-3, betas=(0.9, 0.999))

        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.configure_optimizer(self.optimizer_class, self.optimizer_kwargs)

        self.scheduler_class = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.configure_scheduler(self.scheduler_class, self.scheduler_kwargs)

        self.to(self.device)

    def monitor(self, datasets):
        # Prepare a list for each monitor to hold the batches
        results = [[] for _ in self.monitors]
        for (local_X1, local_y1), (local_X2, local_y2) in zip(
            self.data_generator1(datasets[0], shuffle=False),
            self.data_generator2(datasets[1], shuffle=False),
        ):
            for m in self.monitors:
                m.reset()

            local_X = torch.cat((local_X1, local_X2), dim=2)

            output = self.forward_pass(
                local_X,
                record=True,
                cur_batch_size=len(local_X),
            )

            for k, mon in enumerate(self.monitors):
                results[k].append(mon.get_data())

        return [torch.cat(res, dim=0) for res in results]

    def data_generator1(self, dataset, shuffle=True):
        return self.data_generator1_(dataset, shuffle=shuffle)

    def data_generator2(self, dataset, shuffle=True):
        return self.data_generator2_(dataset, shuffle=shuffle)

    def predict(self, datasets, train_mode=False):
        self.train(train_mode)
        print("predicting")
        if type(datasets) in [torch.Tensor, np.ndarray]:
            output = self.forward_pass(datasets, cur_batch_size=len(datasets))
            pred = self.loss_stack.predict(output)
            return pred
        else:
            # self.prepare_data(data)
            pred = []
            for (local_X1, local_y1), (local_X2, local_y2) in zip(
                self.data_generator1(datasets[0], shuffle=False),
                self.data_generator2(datasets[1], shuffle=False),
            ):
                local_X = torch.cat((local_X1, local_X2), dim=2)
                data_local = local_X.to(self.device)
                output = self.forward_pass(data_local, cur_batch_size=len(local_X))
                pred.append(self.loss_stack.predict(output).detach().cpu())
            return torch.cat(pred, dim=0)

    def train_epoch(self, dataset, shuffle=True):
        self.train(True)
        # self.prepare_data(dataset)
        metrics = []
        for (local_X1, local_y1), (local_X2, local_y2) in zip(
            self.data_generator1(dataset[0], shuffle=False),
            self.data_generator2(dataset[1], shuffle=False),
        ):
            local_X = torch.cat((local_X1, local_X2), dim=2)
            local_y1 = local_y1.unsqueeze(1)
            local_y2 = local_y2.unsqueeze(1)
            local_y = torch.cat((local_y1, local_y2), dim=1)

            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            # split output into parts corresponding to the first and second dataset
            output = torch.split(output, self.nb_outputs // 2, 2)
            total_loss = self.get_total_loss(output, local_y)

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            total_loss.backward()

            self.optimizer_instance.step()
            self.apply_constraints()

        return np.mean(np.array(metrics), axis=0)

    def evaluate(self, dataset, train_mode=False, one_batch=False):
        self.train(train_mode)
        # self.prepare_data(test_dataset)
        metrics = []
        for (local_X1, local_y1), (local_X2, local_y2) in zip(
            self.data_generator1(dataset[0], shuffle=False),
            self.data_generator2(dataset[1], shuffle=False),
        ):
            local_X = torch.cat((local_X1, local_X2), dim=2)
            local_y1 = local_y1.unsqueeze(1)
            local_y2 = local_y2.unsqueeze(1)
            local_y = torch.cat((local_y1, local_y2), dim=1)

            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            # split output into parts corresponding to the first and second dataset
            output = torch.split(output, self.nb_outputs // 2, 2)
            total_loss = self.get_total_loss(output, local_y)
            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )
            if one_batch:
                break

        return np.mean(np.array(metrics), axis=0)

