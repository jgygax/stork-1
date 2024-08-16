import torch
import numpy as np
import time

import stork.nodes.base
from stork.models.base import RecurrentSpikingModel
from stork.loss_stacks import TemporalCrossEntropyReadoutStack
from stork.generators import StandardGenerator


class AnnealingModel(RecurrentSpikingModel):
    def __init__(
        self,
        batch_size,
        nb_time_steps,
        nb_inputs,
        device=torch.device("cpu"),
        dtype=torch.float,
        sparse_input=False,
    ):
        super(RecurrentSpikingModel, self).__init__(
            batch_size, nb_time_steps, nb_inputs, device, dtype, sparse_input
        )

    def configure(
        self,
        input,
        output,
        loss_stack=None,
        optimizer=None,
        optimizer_kwargs=None,
        scheduler=None,
        scheduler_kwargs=None,
        generator=None,
        time_step=1e-3,
        wandb=None,
        # added annealing options
        anneal_interval=0,
        anneal_step=0,
        anneal_start=0,
    ):
        self.input_group = input
        self.output_group = output
        self.time_step = time_step
        self.wandb = wandb

        # annealing options
        self.anneal_interval = anneal_interval
        self.anneal_step = anneal_step
        self.anneal_start = anneal_start

        if loss_stack is not None:
            self.loss_stack = loss_stack
        else:
            self.loss_stack = TemporalCrossEntropyReadoutStack()

        if generator is None:
            self.data_generator_ = StandardGenerator()
        else:
            self.data_generator_ = generator

    def anneal_beta(self, ep, offset=0):
        """
        go through all spiking nonlinearities, change the betas and apply them again. This does not anneal escape noise parameters other than beta.
        """
        for g in range(1, len(self.groups) - 1):
            try:
                beta = self.groups[g].act_fn.surrogate_params["beta"]
                print("beta", beta)
                self.groups[g].act_fn.surrogate_params["beta"] = beta + self.anneal_step
                if "beta" in self.groups[g].act_fn.escape_noise_params.keys():
                    ebeta = self.groups[g].act_fn.escape_noise_params["beta"]
                    print("escape beta", ebeta)
                    self.groups[g].act_fn.escape_noise_params["beta"] = (
                        ebeta + self.anneal_step
                    )
                self.groups[g].spk_nl = self.groups[g].act_fn.apply
                print(
                    "annealed",
                    self.groups[g].act_fn.surrogate_params,
                    self.groups[g].act_fn.escape_noise_params,
                )
                self.wandb.log(
                    {"beta": beta, "noise beta": ebeta}, step=ep + offset + 2
                )

            except Exception as e:
                print(e)
                beta = self.groups[g].act_fn.beta
                print("beta", beta)
                self.groups[g].act_fn.beta = beta + self.anneal_step
                self.groups[g].spk_nl = self.groups[g].act_fn.apply
                print("annealed", self.groups[g].act_fn.beta)
                self.wandb.log({"beta": beta}, step=ep + offset + 2)

    def fit(
        self, dataset, nb_epochs=10, verbose=True, shuffle=True, wandb=None, anneal=True
    ):
        self.hist = []
        self.wall_clock_time = []
        self.train()
        for ep in range(nb_epochs):
            t_start = time.time()
            ret = self.train_epoch(dataset, shuffle=shuffle)
            self.hist.append(ret)

            if self.wandb is not None:
                self.wandb.log(
                    {key: value for (key, value) in zip(self.get_metric_names(), ret)}
                )

            if verbose:
                t_iter = time.time() - t_start
                self.wall_clock_time.append(t_iter)
                print(
                    "%02i %s t_iter=%.2f" % (ep, self.get_metrics_string(ret), t_iter)
                )
            if anneal:
                # TOO: check wheter the offset is needed
                if ep >= self.anneal_start:
                    if (ep - self.anneal_start) % self.anneal_interval == 0:
                        self.anneal_beta(ep)

        self.fit_runs.append(self.hist)
        history = self.get_metrics_history_dict(np.array(self.hist))
        return history

    def fit_validate(
        self,
        dataset,
        valid_dataset,
        nb_epochs=10,
        verbose=True,
        wandb=None,
        anneal=True,
    ):
        self.hist_train = []
        self.hist_valid = []
        self.wall_clock_time = []
        for ep in range(nb_epochs):
            t_start = time.time()
            self.train()
            ret_train = self.train_epoch(dataset)

            self.train(False)
            ret_valid = self.evaluate(valid_dataset)
            self.hist_train.append(ret_train)
            self.hist_valid.append(ret_valid)

            if self.wandb is not None:
                self.wandb.log(
                    {
                        key: value
                        for (key, value) in zip(
                            self.get_metric_names()
                            + self.get_metric_names(prefix="val_"),
                            ret_train.tolist() + ret_valid.tolist(),
                        )
                    }
                )

            if verbose:
                t_iter = time.time() - t_start
                self.wall_clock_time.append(t_iter)
                print(
                    "%02i %s --%s t_iter=%.2f"
                    % (
                        ep,
                        self.get_metrics_string(ret_train),
                        self.get_metrics_string(ret_valid, prefix="val_"),
                        t_iter,
                    )
                )
            if anneal:
                # TOO: check wheter the offset is correctly used
                if ep >= self.anneal_start:
                    if (ep - self.anneal_start) % self.anneal_interval == 0:
                        self.anneal_beta(ep)

        self.hist = np.concatenate(
            (np.array(self.hist_train), np.array(self.hist_valid))
        )
        self.fit_runs.append(self.hist)
        dict1 = self.get_metrics_history_dict(np.array(self.hist_train), prefix="")
        dict2 = self.get_metrics_history_dict(np.array(self.hist_valid), prefix="val_")
        history = {**dict1, **dict2}
        return history
