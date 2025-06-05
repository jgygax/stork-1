import torch
import numpy as np


class Monitor:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError


class SpikeMonitor(Monitor):
    """Records spikes in sparse RAS format

    Args:
        group: The group to record from

    Returns:
        argwhere of out sequence
    """

    def __init__(self, group, name="SpikeMonitor"):
        super().__init__()
        self.group = group
        self.batch_count_ = 0
        self.name = name

    def reset(self):
        pass

    def execute(self):
        pass

    def get_data(self):
        out = self.group.get_out_sequence().detach().cpu()
        tmp = torch.nonzero(out)
        tmp[:, 0] += self.batch_count_
        self.batch_count_ += out.shape[0]
        return tmp


class StateMonitor(Monitor):
    """Records the state of a neuron group over time

    Args:
        group: The group to record from
        key: The name of the state
    """

    def __init__(self, group, key, subset=None, name="StateMonitor"):
        super().__init__()
        self.group = group
        self.key = key
        self.subset = subset
        self.name = name

    def reset(self):
        self.data = []

    def execute(self):
        if self.subset is not None:
            self.data.append(self.group.states[self.key][:, self.subset].detach().cpu())
        else:
            self.data.append(self.group.states[self.key].detach().cpu())

    def get_data(self):
        return torch.stack(self.data, dim=1)


class SpikeCountMonitor(Monitor):
    """Counts number of spikes (sum over time in get_out_sequence() for each neuron)

    Args:
        group: The group to record from

    Returns:
        A tensor with spike counts for each input and neuron
    """

    def __init__(self, group, name="SpikeCountMonitor"):
        super().__init__()
        self.group = group
        self.name = name

    def reset(self):
        pass

    def execute(self):
        pass

    def get_data(self):
        return torch.sum(self.group.get_out_sequence().detach().cpu(), dim=1)


class PopulationSpikeCountMonitor(Monitor):
    """Counts total number of spikes (sum over time in get_out_sequence() for the group)

    Args:
        group: The group to record from

    Returns:
        A tensor with spike counts for each input and neuron
    """

    def __init__(self, group, name="PopulationSpikeCountMonitor"):
        super().__init__()
        self.group = group
        self.name = name

    def reset(self):
        self.data = []

    def execute(self):
        pass

    def get_data(self):
        s1 = torch.sum(self.group.get_out_sequence().detach().cpu(), dim=1)
        return torch.mean(s1, dim=1)


class PopulationFiringRateMonitor(Monitor):
    """Monitors population firing rate (nr of spikes / nr of neurons for every timestep)

    Args:
        group: The group to record from

    Returns:
        A tensor with population firing rate for each input and timestep
    """

    def __init__(self, group, name="PopulationFiringRateMonitor"):
        super().__init__()
        self.group = group
        self.name = name

    def reset(self):
        self.data = []

    def execute(self):
        pass

    def get_data(self):
        s1 = self.group.get_out_sequence().detach().cpu()
        s1 = s1.reshape(s1.shape[0], s1.shape[1], self.group.nb_units)
        return torch.sum(s1, dim=-1) / self.group.nb_units


class MeanVarianceMonitor(Monitor):
    """Measures mean and variance of input

    Args:
        group: The group to record from
        state (string): State variable to monitor (Monitors mean and variance of a state variable)


    Returns:
        A tensors with mean and variance for each neuron/state along the last dim
    """

    def __init__(self, group, state="input", name="MeanVarianceMonitor"):
        super().__init__()
        self.group = group
        self.key = state
        self.name = name

    def reset(self):
        self.s = 0
        self.s2 = 0
        self.c = 0

    def execute(self):
        tns = self.group.states[self.key]
        self.s += tns.detach().cpu()
        self.s2 += torch.square(tns).detach().cpu()
        self.c += 1

    def get_data(self):
        mean = self.s / self.c
        var = self.s2 / self.c - mean**2
        return torch.stack((mean, var), len(mean.shape))


class GradientMonitor(Monitor):
    """Records the gradients (weight.grad)

    Args:
        target: The tensor or nn.Module to record from
                (usually a stork.connection.op object)
                Needs to have a .weight argument
    """

    def __init__(self, target, name="GradientMonitor"):
        super().__init__()
        self.target = target
        self.name = name

    def reset(self):
        pass

    def set_hook(self):
        """
        Sets the backward hook
        """
        pass

    def remove_hook(self):
        pass

    def execute(self):
        pass

    def get_data(self):
        # unsqueeze so that the output from the monitor is [batch_nr x weightmatrix-dims]
        return self.target.weight.grad.detach().cpu().abs().unsqueeze(0)


class GradientOutputMonitor(GradientMonitor):
    """Records the gradients wrt the neuronal output
        computed in the backward pass

    Args:
        target: The tensor or nn.Module to record from
                (usually a stork.connection.op object)
    """

    def __init__(self, target, name="GradientOutputMonitor"):
        super().__init__(target)
        self.count = 0
        self.sum = 0
        self.name = name

    def set_hook(self):
        """
        Sets the backward hook
        """
        self.hook = self.target.register_full_backward_hook(self.grab_gradient)

    def remove_hook(self):
        self.hook.remove()

    def grab_gradient(self, module, grad_input, grad_output):
        mean_grad = grad_output[0].detach().cpu().abs()
        self.sum += mean_grad
        self.count += 1

    def reset(self):
        self.count = 0
        self.sum = 0

    def execute(self):
        pass

    def get_data(self):
        return self.sum / self.count


##################################################################################################
# NEW MONITORS (TODO: still to be tested)
##################################################################################################


class BalanceMonitor(Monitor):
    """Records balance of a neuron group over time

    Args:
        group: The group to record from
        key: The name of the state
    """

    def __init__(
        self,
        group,
        key_exc="syne",
        key_inh="syni",
        scaling=True,
        subset=None,
        name="StateMonitor",
        eps=1e-8,
    ):
        super().__init__()
        self.group = group
        self.key_exc = key_exc
        self.key_inh = key_inh
        self.scaling = scaling
        self.subset = subset
        self.name = name
        self.eps = eps  # small value to avoid division by zero

    def reset(self):
        self.data_exc = []
        self.data_inh = []

    def get_balance(self, syne, syni):
        num = (syni - syne) ** 2
        denom = (syne + syni) ** 2 + self.eps
        return torch.tensor(num / denom)

    def execute(self):
        if self.subset is not None:
            exc = self.group.states[self.key_exc][:, self.subset]
            inh = self.group.states[self.key_inh][:, self.subset]
        else:
            exc = self.group.states[self.key_exc]
            inh = self.group.states[self.key_inh]

        self.data_exc.append(exc.detach().cpu())
        self.data_inh.append(inh.detach().cpu())

    def scl_data(self):
        scl = torch.max(
            torch.tensor([torch.max(self.data_exc), torch.max(self.data_inh)])
        )
        self.data_exc /= scl
        self.data_inh /= scl

    # abstract method to be implemented by subclasses
    def get_data(self):
        pass


class PreciseBalanceMonitor(BalanceMonitor):
    """Records precise balance of a neuron group over time

    Args:
        group: The group to record from
        key_exc: The name of the excitatory state
        key_inh: The name of the inhibitory state
        scaling: Whether to scale the states
        subset: A subset of neurons to record from
        name: The name of the monitor
        thr: Threshold for balance calculation
    """

    def __init__(self, thr, **kwargs):
        super().__init__(**kwargs)
        self.thr = thr

    def get_data(self):
        self.data_exc = torch.stack(self.data_exc, dim=1)
        self.data_inh = torch.stack(self.data_inh, dim=1)

        if self.scaling:
            self.scl_data()

        self.data_exc = self.data_exc.flatten()
        self.data_inh = self.data_inh.flatten()

        if self.thr is not None:
            mask = self.data_exc > self.thr & self.data_inh > self.thr
            self.data_exc = self.data_exc[mask]
            self.data_inh = self.data_inh[mask]

        return self.get_balance(self.data_exc, self.data_inh)


class DetailedBalanceMonitor(BalanceMonitor):
    """Records detailed balance of a neuron group (averaged over time)

    Args:
        group: The group to record from
        key_exc: The name of the excitatory state
        key_inh: The name of the inhibitory state
        scaling: Whether to scale the states
        subset: A subset of neurons to record from
        name: The name of the monitor
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_data(self):
        self.data_exc = torch.stack(self.data_exc, dim=1)
        self.data_inh = torch.stack(self.data_inh, dim=1)

        if self.scaling:
            self.scl_data()

        self.data_exc = self.data_exc.mean(dim=1).flatten()
        self.data_inh = self.data_inh.mean(dim=1).flatten()

        return self.get_balance(self.data_exc, self.data_inh)


class TightBalanceMonitor(BalanceMonitor):
    """Records tight balance of a neuron group (averaged over stimuli)

    Args:
        group: The group to record from
        key_exc: The name of the excitatory state
        key_inh: The name of the inhibitory state
        scaling: Whether to scale the states
        subset: A subset of neurons to record from
        name: The name of the monitor
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_data(self):
        self.data_exc = torch.stack(self.data_exc, dim=1)
        self.data_inh = torch.stack(self.data_inh, dim=1)

        if self.scaling:
            self.scl_data()

        self.data_exc = self.data_exc.mean(dim=0).flatten()
        self.data_inh = self.data_inh.mean(dim=0).flatten()

        return self.get_balance(self.data_exc, self.data_inh)


class GlobalBalanceMonitor(BalanceMonitor):
    """Records global balance of a neuron group

    Args:
        group: The group to record from
        key_exc: The name of the excitatory state
        key_inh: The name of the inhibitory state
        scaling: Whether to scale the states
        subset: A subset of neurons to record from
        name: The name of the monitor
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_data(self):
        self.data_exc = torch.stack(self.data_exc, dim=1)
        self.data_inh = torch.stack(self.data_inh, dim=1)

        if self.scaling:
            self.scl_data()

        self.data_exc = self.data_exc.mean(dim=(0, 1)).flatten()
        self.data_inh = self.data_inh.mean(dim=(0, 1)).flatten()

        return self.get_balance(self.data_exc, self.data_inh)


class StdevPopulationFiringRateMonitor(Monitor):
    """Monitors the standard deviation of the population firing rate (nr of spikes / nr of neurons for every timestep)

    Args:
        group: The group to record from

    Returns:
        A tensor with the standard deviation of the population firing rate for each input and timestep
    """

    def __init__(self, group, name="stdevPopulationFiringRateMonitor"):
        super().__init__()
        self.group = group
        self.name = name

    def reset(self):
        self.data = []

    def execute(self):
        pass

    def get_data(self):
        s1 = self.group.get_out_sequence().detach().cpu()
        s1 = s1.reshape(s1.shape[0], s1.shape[1], self.group.nb_units)
        return torch.std(s1, dim=-1)


class SynchronyMonitor(Monitor):
    """Measures the synchrony of a group of neurons

    Args:
        group: The group to record from
    """

    def __init__(self, group, bin_steps=1, name="SynchronyMonitor"):
        super().__init__()
        self.group = group
        self.bin_steps = bin_steps
        self.name = name

    def reset(self):
        self.data = []

    def execute(self):
        pass

    def get_data(self):
        g = self.group.get_out_sequence().detach().cpu()
        # get the moving average of the spikes with window size bin_steps
        g = torch.nn.functional.avg_pool1d(g, self.bin_steps, stride=1).bool().float()
        m = torch.mean(g, dim=-1)
        synchrony = torch.std(m, dim=-1)
        return synchrony


class IrregularityMonitor(Monitor):
    """Measures the regularity of a group of neurons

    Args:
        group: The group to record from
    """

    def __init__(self, group, bin_steps=1, name="IrregularityMonitor"):
        super().__init__()
        self.group = group
        self.bin_steps = bin_steps
        self.name = name

    def reset(self):
        self.data = []

    def execute(self):
        pass

    def get_isi(self, inp):
        isis = []
        for neuron in inp.T:
            isis += np.diff(np.where(neuron)[0]).tolist()
        # remove all isis that are smaller than bin_steps
        isis = [x for x in isis if x >= self.bin_steps]
        return torch.tensor(isis, dtype=torch.float32)

    def get_cv_isi(self, isis):
        return torch.std(isis) / torch.mean(isis)

    def get_data(self):
        g = self.group.get_out_sequence().detach().cpu()
        # get the moving average of the spikes with window size bin_steps
        g = torch.nn.functional.avg_pool1d(g, self.bin_steps, stride=1)

        cvisis = []
        for sample in g:
            isis = self.get_isi(sample)
            cv_isi = self.get_cv_isi(isis)
            cvisis.append(cv_isi)
        cvisis = torch.tensor(cvisis, dtype=torch.float32)
        return cvisis
