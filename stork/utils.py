import gzip
import pickle
import torch
import string
import random
import time
import numpy as np


def get_random_string(string_length=5):
    """Generates a random string of fixed length"""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(string_length))


def get_basepath(dir=".", prefix="default", salt_length=5):
    """Returns pre-formatted and time stamped basepath given a base directory and file prefix."""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if salt_length:
        salt = get_random_string(salt_length)
        basepath = "%s/%s-%s-%s" % (dir, prefix, timestr, salt)
    else:
        basepath = "%s/%s-%s" % (dir, prefix, timestr)
    return basepath


def write_to_file(data, filename):
    """Writes an object/dataset to zipped pickle.

    Args:
        data: the (data) object
        filename (str): the filename to write to
    """
    fp = gzip.open("%s" % filename, "wb")
    pickle.dump(data, fp)
    fp.close()


def load_from_file(filename):
    """Loads an object/dataset from a zipped pickle."""
    fp = gzip.open("%s" % filename, "r")
    data = pickle.load(fp)
    fp.close()
    return data


def to_sparse(x):
    """converts dense tensor x to sparse format"""

    indices = torch.nonzero(x)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse.FloatTensor(indices, values, x.size(), device=x.device)


def get_lif_kernel(tau_mem=20e-3, tau_syn=10e-3, dt=1e-3):
    """Computes the linear filter kernel of a simple LIF neuron with exponential current-based synapses.

    Args:
        tau_mem: The membrane time constant
        tau_syn: The synaptic time constant
        dt: The timestep size

    Returns:
        Array of length 10x of the longest time constant containing the filter kernel

    """
    tau_max = np.max((tau_mem, tau_syn))
    ts = np.arange(0, int(tau_max * 10 / dt)) * dt
    n = len(ts)
    kernel = np.empty(n)
    I = 1.0  # Initialize current variable for single spike input
    U = 0.0
    dcy1 = np.exp(-dt / tau_mem)
    dcy2 = np.exp(-dt / tau_syn)
    for i, t in enumerate(ts):
        kernel[i] = U
        U = dcy1 * U + (1.0 - dcy1) * I
        I *= dcy2
    return kernel


def get_2lif_kernel(tau_mem=20e-3, tau_syn1=5e-3, tau_syn2=100e-3, dt=1e-3):
    """Computes the linear filter kernel of a simple LIF neuron with exponential current-based synapses.

    Args:
        tau_mem: The membrane time constant
        tau_syn1: The first synaptic time constant (ampa)
        tau_syn2: The second synaptic time constant (nmda)
        dt: The timestep size

    Returns:
        Array of length 10x of the longest time constant containing the filter kernel

    """
    tau_max = np.max((tau_mem, tau_syn1, tau_syn2))
    ts = np.arange(0, int(tau_max * 10 / dt)) * dt
    n = len(ts)
    kernel = np.empty(n)

    I1 = 1  # Initialize current variable for single spike input
    I2 = 0
    U = 0

    dcym = np.exp(-dt / tau_mem)
    dcy1 = np.exp(-dt / tau_syn1)
    dcy2 = np.exp(-dt / tau_syn2)

    for i, t in enumerate(ts):
        kernel[i] = U
        U = dcym * U + (1 - dcym) * (I1 + I2) / 2
        I1 *= dcy1
        I2 *= dcy2 + (1 - dcy2) * I1
    return kernel


def convlayer_size(nb_inputs, kernel_size, padding, stride):
    """
    Calculates output size of convolutional layer
    """
    res = ((np.array(nb_inputs) - kernel_size + 2 * padding) / stride) + 1
    return res
