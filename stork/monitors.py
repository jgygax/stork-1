import torch


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


class BalanceMonitor(Monitor):
    """Records the balance index of a neuron group over time.
    This is only applicable to Dalian groups

    Args:
        group: The group to record from
        key_exc: The name of the excitatory state
        key_inh: The name of the inhibitory state
    """

    def __init__(
        self,
        group,
        key_exc="exc",
        key_inh="inh",
        subset=None,
        eps=1e-10,
        name="BalanceMonitor",
    ):
        super().__init__()
        self.group = group
        self.key_exc = key_exc
        self.key_inh = key_inh
        self.subset = subset
        self.eps = eps
        self.name = name

    def reset(self):
        self.data_exc = []
        self.data_inh = []

    def execute(self):
        if self.subset is not None:
            self.data_exc.append(
                self.group.states[self.key_exc][:, self.subset].detach().cpu()
            )
            self.data_inh.append(
                self.group.states[self.key_inh][:, self.subset].detach().cpu()
            )
        else:
            self.data_exc.append(self.group.states[self.key_exc].detach().cpu())
            self.data_inh.append(self.group.states[self.key_inh].detach().cpu())

    def get_data(self):

        exc = torch.stack(self.data_exc, dim=1)
        inh = torch.stack(self.data_inh, dim=1)

        diff = exc - inh
        summe = exc + inh
        return torch.multiply(diff, diff) / (torch.multiply(summe, summe) + self.eps)


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
        return torch.mean(s1)


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
