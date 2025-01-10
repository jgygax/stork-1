import torch
import numpy as np

import stork.nodes.base
from stork.models.base import RecurrentSpikingModel
from stork import generators
from stork import loss_stacks


class DoubleLossRecSpikingModel(RecurrentSpikingModel):
    def __init__(
        self,
        batch_size,
        nb_time_steps,
        nb_inputs,
        nb_outputs_AE,
        nb_outputs_class,
        device=torch.device("cpu"),
        dtype=torch.float,
        sparse_input=False,
    ):
        super(RecurrentSpikingModel, self).__init__()
        self.batch_size = batch_size
        self.nb_time_steps = nb_time_steps
        self.nb_inputs = nb_inputs
        self.nb_outputs_AE = nb_outputs_AE
        self.nb_outputs_class = nb_outputs_class

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
        output_AE,
        output_class,
        loss_AE=None,
        AE_fr_loss=False,
        loss_class=None,
        optimizer=None,
        optimizer_kwargs=None,
        scheduler=None,
        scheduler_kwargs=None,
        generator=None,
        time_step=1e-3,
        wandb=None,
    ):
        self.input_group = input
        self.output_group_AE = output_AE
        self.output_group_class = output_class
        self.time_step = time_step
        self.wandb = wandb
        self.AE_fr_loss = AE_fr_loss

        if loss_AE is not None:
            self.loss_AE = loss_AE
        else:
            self.loss_AE = loss_stacks.FiringRateReconstructionLoss()
        if loss_class is not None:
            self.loss_class = loss_class
        else:
            self.loss_class = loss_stacks.SumOfSoftmaxCrossEntropy()

        if generator is None:
            self.data_generator_ = generators.StandardGenerator()
        else:
            self.data_generator_ = generator

        # configure data generator
        self.data_generator_.configure(
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

    def monitor(self, dataset):
        self.prepare_data(dataset)

        # Prepare a list for each monitor to hold the batches
        results = {mon.name + "-" + str(mon.group.name): [] for mon in self.monitors}
        for local_X, local_y in self.data_generator(dataset, shuffle=False):
            for m in self.monitors:
                m.reset()

            output = self.forward_pass(
                local_X, record=True, cur_batch_size=len(local_X)
            )

            for k, mon in enumerate(self.monitors):
                results[mon.name + "-" + str(mon.group.name)].append(mon.get_data())

        for k, v in results.items():
            results[k] = torch.cat(v, dim=0)
        return results  # [torch.cat(res, dim=0) for res in results]

    def predict(self, dataset, train_mode=False):
        self.train(train_mode)
        print("predicting")
        if type(dataset) in [torch.Tensor, np.ndarray]:
            output = self.forward_pass(dataset, cur_batch_size=len(dataset))
            pred = self.loss_stack.predict(output)
            return pred
        else:
            # self.prepare_data(data)
            pred = []
            for local_X, (local_y_AE, local_y_class) in self.data_generator(
                dataset, shuffle=False
            ):
                data_local = local_X.to(self.device)
                output = self.forward_pass(data_local, cur_batch_size=len(local_X))
                pred.append(self.loss_stack.predict(output).detach().cpu())
            return torch.cat(pred, dim=0)

    def train_epoch(self, dataset, shuffle=True):
        self.train(True)
        # self.prepare_data(dataset)
        metrics = []
        for local_X, (local_y_AE, local_y_class) in self.data_generator(
            dataset, shuffle=False
        ):

            output_AE, output_class = self.forward_pass(
                local_X, cur_batch_size=len(local_X)
            )
            # split output into parts corresponding to the first and second dataset
            total_loss = self.get_total_loss(
                output_AE,
                self.input_group.get_out_sequence().detach(),
                output_class,
                local_y_class,
            )

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()]
                + self.loss_AE.metrics
                + self.loss_class.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            total_loss.backward()

            self.optimizer_instance.step()
            self.apply_constraints()

        return np.mean(np.array(metrics), axis=0)

    def run(self, x_batch, cur_batch_size=None, record=False):
        if cur_batch_size is None:
            cur_batch_size = len(x_batch)
        self.reset_states(cur_batch_size)
        self.input_group.feed_data(x_batch)
        for t in range(self.nb_time_steps):
            stork.nodes.base.CellGroup.clk = t
            self.evolve_all()
            self.propagate_all()
            self.execute_all()
            if record:
                self.monitor_all()
        self.out_AE = self.output_group_AE.get_out_sequence()
        self.out_class = self.output_group_class.get_out_sequence()
        return self.out_AE, self.out_class

    def get_total_loss(
        self, output_AE, target_AE, output_class, target_class, regularized=True
    ):
        target_AE = target_AE.to(self.device)
        target_class = target_class.to(self.device)

        if self.AE_fr_loss:
            self.out_loss = self.loss_AE(
                output_AE, torch.sum(target_AE, dim=1)
            ) + self.loss_class(output_class, target_class)
        else:
            self.out_loss = self.loss_AE(output_AE, target_AE) + self.loss_class(
                output_class, target_class
            )

        if regularized:
            self.reg_loss = self.compute_regularizer_losses()
            total_loss = self.out_loss + self.reg_loss
        else:
            total_loss = self.out_loss

        return total_loss

    def get_metric_names(self, prefix="", postfix=""):
        metric_names = (
            ["loss", "reg_loss"]
            + self.loss_AE.get_metric_names()
            + self.loss_class.get_metric_names()
        )
        return ["%s%s%s" % (prefix, k, postfix) for k in metric_names]

    def evaluate(self, dataset, train_mode=False, two_batches=False):
        self.train(train_mode)
        # self.prepare_data(test_dataset)
        metrics = []
        for i, (local_X, (local_y_AE, local_y_class)) in enumerate(self.data_generator(
            dataset, shuffle=False
        )):
            if two_batches and i == 2:
                break

            output_AE, output_class = self.forward_pass(
                local_X, cur_batch_size=len(local_X)
            )

            # split output into parts corresponding to the first and second dataset
            total_loss = self.get_total_loss(
                output_AE,
                self.input_group.get_out_sequence().detach(),
                output_class,
                local_y_class,
            )
            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()]
                + self.loss_AE.metrics
                + self.loss_class.metrics
            )

        return np.mean(np.array(metrics), axis=0)
