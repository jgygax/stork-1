import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stork import datasets
from scipy.stats import linregress


class Plotter:

    def __init__(self):
        pass

    def add_scalebar(
        self,
        ax,
        extent=(1.0, 0.0),
        pos=(0.0, 0.0),
        off=(0.0, 0.0),
        label=None,
        **kwargs,
    ):
        scale = np.concatenate((np.diff(ax.get_xlim()), np.diff(ax.get_ylim())))
        x1, y1 = np.array(pos) * scale
        x2, y2 = np.array((x1, y1)) + np.array(extent)
        xt, yt = (
            np.array((np.mean((x1, x2)), np.mean((y1, y2)))) + np.array(off) * scale
        )
        ax.plot((x1, x2), (y1, y2), color="black", lw=5)
        if label:
            ax.text(xt, yt, label, **kwargs)

    def add_xscalebar(
        self, ax, length, label=None, pos=(0.0, -0.1), off=(0.0, -0.07), **kwargs
    ):
        self.add_scalebar(
            ax,
            label=label,
            extent=(length, 0.0),
            pos=pos,
            off=off,
            verticalalignment="top",
            horizontalalignment="center",
            **kwargs,
        )

    def add_yscalebar(
        self, ax, length, label=None, pos=(-0.1, 0.0), off=(-0.07, 0.0), **kwargs
    ):
        self.add_scalebar(
            ax,
            label=label,
            extent=(0.0, length),
            pos=pos,
            off=off,
            verticalalignment="center",
            horizontalalignment="left",
            rotation=90,
            **kwargs,
        )

    def dense2scatter_plot(
        self,
        ax,
        dense,
        point_size=5,
        alpha=1.0,
        marker=".",
        time_step=1e-3,
        jitter=None,
        double=False,
        color_list=["black", "black"],
        **kwargs,
    ):
        n = dense.shape[1] // 2
        if double:
            ras0 = datasets.dense2ras(dense[:, :n], time_step)
            ras1 = datasets.dense2ras(dense[:, n:], time_step)
            ras = [ras0, ras1]
        else:
            ras = [datasets.dense2ras(dense, time_step)]
        for r, c in zip(ras, color_list):
            if len(r):
                noise = np.zeros(r[:, 0].shape)
                if jitter is not None:
                    noise = jitter * np.random.randn(*r[:, 0].shape)
                ax.scatter(
                    r[:, 0] + noise,
                    r[:, 1],
                    s=point_size,
                    alpha=alpha,
                    marker=marker,
                    color=c,
                    **kwargs,
                )

    def save_plots(self, fileprefix, extensions=["pdf", "png"], dpi=300):
        """Apply savefig function to multiple extensions"""
        for ext in extensions:
            plt.savefig("%s.%s" % (fileprefix, ext), dpi=dpi, bbox_inches="tight")

    def turn_axis_off(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    def plot_spike_raster(
        self,
        ax,
        data,
        color,
        point_size,
        marker,
        point_alpha,
        set_axis=False,
    ):
        ax.scatter(
            np.where(data)[0],
            np.where(data)[1],
            s=point_size,
            marker=marker,
            color=color,
            alpha=point_alpha,
        )
        # invert y-axis
        ax.invert_yaxis()

        if set_axis:
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        else:
            self.turn_axis_off(ax)


class ActivityPlotter(Plotter):
    def __init__(
        self,
        model,
        data,
        color_classes=True,
        plot_label=False,
        label_color="crimson",
        color="k",
        marker=".",
        set_axis=False,
        nb_samples=2,
        samples=None,
        figsize=(5, 5),
        dpi=250,
        point_size=5,
        point_alpha=1,
        pal=sns.color_palette("muted", n_colors=20),
        bg_col="#AAAAAA",
        scale_spike_rasters=4,
        readout_layers=[-1],
        batch_size=None,
        scale_input=1,
    ):
        super().__init__()
        self.color_classes = color_classes
        self.plot_label = plot_label
        self.label_color = label_color
        self.scale_spike_rasters = scale_spike_rasters
        self.color = color
        self.point_size = point_size
        self.marker = marker
        self.point_alpha = point_alpha
        self.set_axis = set_axis
        self.batch_size = batch_size
        self.scale_input = scale_input

        self.nb_samples = nb_samples
        self.samples = samples
        self.figsize = figsize
        self.dpi = dpi
        self.pal = pal
        self.bg_col = bg_col

        self.model = model
        self.data = data
        self.readout_layers = readout_layers

        nb_groups = len(self.model.groups)
        self.nb_spiking_groups = nb_groups - len(self.readout_layers)

    def get_activities(self):

        # Run model once and get activities
        scores = self.model.evaluate(self.data, two_batches=True).tolist()
        activities = [
            g.get_flattened_out_sequence().detach().cpu().numpy()
            for g in self.model.groups
        ]

        if self.batch_size is None:
            self.batch_size = len(activities[0])

        return activities

    def get_height_ratios(self):
        nb_groups = len(self.model.groups)

        nb_total_units = np.sum(
            [self.model.groups[g].nb_units for g in range(self.nb_spiking_groups)]
        )
        hr = [
            self.scale_spike_rasters * self.model.groups[g].nb_units / nb_total_units
            for g in range(self.nb_spiking_groups)
        ] + [1] * len(self.readout_layers)
        hr[0] *= self.scale_input
        hr = list(reversed(hr))  # since we are plotting from bottom to top
        return hr

    def plot_input(
        self,
        ax,
        data,
        color,
    ):
        self.plot_spike_raster(
            ax,
            data,
            color,
            self.point_size,
            self.marker,
            self.point_alpha,
            set_axis=True,
        )

    def plot_hidden(
        self,
        ax,
        data,
    ):
        self.plot_spike_raster(
            ax,
            data,
            self.color,
            self.point_size,
            self.marker,
            self.point_alpha,
            self.set_axis,
        )

    def plot_readout(self, ax, data, label, pal, bg_col):
        if self.tuple_label:
            label = label[1]
        for line_index, ro_line in enumerate(np.transpose(data)):
            if self.color_classes and line_index == label:
                ax.plot(ro_line, color=pal[line_index])
            else:
                ax.plot(ro_line, color=bg_col, zorder=-5, alpha=0.5)

        if self.plot_label:
            ax.plot(label, color=self.label_color, zorder=-10)

        self.turn_axis_off(ax)

    def plot_activity(
        self,
    ):
        print("plotting")
        activities = self.get_activities()
        nb_groups = len(self.model.groups)

        hr = self.get_height_ratios()

        fig, ax = plt.subplots(
            len(hr),
            self.nb_samples,
            figsize=self.figsize,
            dpi=self.dpi,
            sharex="row",
            sharey="row",
            gridspec_kw={"height_ratios": hr},
        )

        if self.nb_samples == 1:
            ax = np.array([ax]).T

        sns.despine()

        if self.samples is None:
            self.samples = list(range(self.nb_samples))

        ##############################################################################################
        # Plotting
        ##############################################################################################

        ylims = {g: (float("inf"), float("-inf")) for g in range(nb_groups)}
        for i, s in enumerate(self.samples):

            self.tuple_label = False

            # plot and color input spikes
            if self.color_classes:
                try:
                    c = self.pal[self.data[s + self.batch_size][1]]
                except:
                    c = self.pal[self.data[s + self.batch_size][1][1]]
                    self.tuple_label = True
            else:
                c = "darkgray"

            self.plot_input(
                ax[-1, i],
                activities[0][s],
                c,
            )
            ax[-1][0].set_ylabel(self.model.groups[0].name)

            # plot hidden layer spikes
            for g in range(1, self.nb_spiking_groups):
                self.plot_hidden(
                    ax[-(1 + g), i],
                    activities[g][s],
                )
                ax[-(1 + g)][0].set_ylabel(self.model.groups[g].name)

                if i == 0:
                    ax[-(1 + g)][i].set_yticks([])

            # plot readout
            for g in range(self.nb_spiking_groups, nb_groups):
                self.plot_readout(
                    ax[-(1 + g), i],
                    activities[g][s],
                    self.data[s + self.batch_size][1],
                    self.pal,
                    self.bg_col,
                )
                ax[-(1 + g), 0].set_ylabel(self.model.groups[g].name)
                ax[-(1 + g)][0].set_yticks([])

                ylims[g] = (
                    min(ylims[g][0], min(activities[g][s].flatten())),
                    max(ylims[g][1], max(activities[g][s].flatten())),
                )

            ax[-1][i].set_xlabel("Time (s)")

        for g in range(self.nb_spiking_groups, nb_groups):
            ax[-(1 + g)][0].set_ylim(ylims[g])

        ax[-1][0].set_ylabel("Input")
        ax[-1][0].set_yticks([])
        ax[-1][0].set_xlim(-3, self.model.nb_time_steps + 3)

        duration = round(self.model.nb_time_steps * self.model.time_step * 10) / 10
        ax[-1][0].set_xticks([0, self.model.nb_time_steps], [0, duration])

        # get ylims of ax[0][0] and set the yticks
        # ylims = ax[0][0].get_ylim()
        # ax[0][0].set_yticks([0, 1])

        return fig, ax


class ReadoutAveraged(ActivityPlotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_readout(self, ax, data, label, pal, bg_col):
        pal = self.label_color
        if self.tuple_label:
            label = label[1]
        for line_index, ro_line in enumerate(np.transpose(data)):
            ax.plot(ro_line, color=pal[line_index])

            x = ro_line.shape[-1]

        if self.plot_label:
            ax.scatter(
                [x] * len(label), label, color=self.label_color, zorder=-10, s=20
            )

        ax.set_xticks([])


class SplitReadoutPlot(ActivityPlotter):
    def __init__(self, nb_splits=2, **kwargs):
        super().__init__(**kwargs)
        self.nb_splits = nb_splits

    def get_height_ratios(self):
        nb_groups = len(self.model.groups)

        nb_total_units = np.sum(
            [self.model.groups[g].nb_units for g in range(self.nb_spiking_groups)]
        )
        hr = [
            self.scale_spike_rasters * self.model.groups[g].nb_units / nb_total_units
            for g in range(self.nb_spiking_groups)
        ] + [1] * len(self.readout_layers) * self.nb_splits
        hr[0] *= self.scale_input
        hr = list(reversed(hr))  # since we are plotting from bottom to top
        return hr

    def plot_readout(self, ax, data, label, pal, bg_col):
        for line_index, ro_line in enumerate(np.transpose(data)):
            ax.plot(ro_line, color=pal[line_index])

        if self.plot_label:
            for line_index, label_line in enumerate(np.transpose(label)):
                ax.plot(
                    label_line, color=pal[line_index], ls="--", alpha=0.75, zorder=-10
                )

        self.turn_axis_off(ax)

    def plot_activity(
        self,
    ):
        print("plotting")
        activities = self.get_activities()
        nb_groups = len(self.model.groups) + self.nb_splits - 1

        hr = self.get_height_ratios()

        fig, ax = plt.subplots(
            len(hr),
            self.nb_samples,
            figsize=self.figsize,
            dpi=self.dpi,
            sharex="row",
            sharey="row",
            gridspec_kw={"height_ratios": hr},
        )

        if self.nb_samples == 1:
            ax = np.array([ax]).T

        sns.despine()

        if self.samples is None:
            self.samples = list(range(self.nb_samples))

        ##############################################################################################
        # Plotting
        ##############################################################################################

        ylims = {g: (float("inf"), float("-inf")) for g in range(nb_groups)}
        for i, s in enumerate(self.samples):

            self.tuple_label = False

            # plot and color input spikes
            if self.color_classes:
                try:
                    c = self.pal[self.data[s + self.batch_size][1]]
                except:
                    c = self.pal[self.data[s + self.batch_size][1][1]]
                    self.tuple_label = True
            else:
                c = "darkgray"

            self.plot_input(
                ax[-1, i],
                activities[0][s],
                c,
            )
            ax[-1][0].set_ylabel(self.model.groups[0].name)

            # plot hidden layer spikes
            for g in range(1, self.nb_spiking_groups):
                self.plot_hidden(
                    ax[-(1 + g), i],
                    activities[g][s],
                )
                ax[-(1 + g)][0].set_ylabel(self.model.groups[g].name)

                if i == 0:
                    ax[-(1 + g)][i].set_yticks([])

            # plot readout
            ro_chunk = int(activities[-1][s].shape[1] / self.nb_splits)
            for split_idx, g in enumerate(range(self.nb_spiking_groups, nb_groups)):
                self.plot_readout(
                    ax[-(1 + g), i],
                    activities[-1][s][
                        :, split_idx * ro_chunk : (split_idx + 1) * ro_chunk
                    ],
                    self.data[s + self.batch_size][1][
                        :, split_idx * ro_chunk : (split_idx + 1) * ro_chunk
                    ],
                    self.pal,
                    self.bg_col,
                )
                if split_idx:
                    ax[-(1 + g), 0].set_ylabel("Class")
                ax[-(1 + g), 0].set_ylabel(self.model.groups[-1].name)
                ax[-(1 + g)][0].set_yticks([])

                ylims[g] = (
                    min(
                        ylims[g][0],
                        min(
                            np.concatenate(
                                (
                                    activities[-1][s].flatten(),
                                    self.data[s + self.batch_size][1].flatten(),
                                )
                            )
                        ),
                    ),
                    max(
                        ylims[g][1],
                        max(
                            np.concatenate(
                                (
                                    activities[-1][s].flatten(),
                                    self.data[s + self.batch_size][1].flatten(),
                                )
                            )
                        ),
                    ),
                )

            ax[-1][i].set_xlabel("Time (s)")

        for g in range(self.nb_spiking_groups, nb_groups):
            ax[-(1 + g)][0].set_ylim(ylims[g])

        ax[-1][0].set_ylabel("Input")
        ax[-1][0].set_yticks([])
        ax[-1][0].set_xlim(-3, self.model.nb_time_steps + 3)

        duration = round(self.model.nb_time_steps * self.model.time_step * 10) / 10
        ax[-1][0].set_xticks([0, self.model.nb_time_steps], [0, duration])

        # get ylims of ax[0][0] and set the yticks
        # ylims = ax[0][0].get_ylim()
        # ax[0][0].set_yticks([0, 1])

        return fig, ax


class CurrentInputActivityPlotter(ActivityPlotter):
    def __init__(self):
        super().__init__()

    def get_height_ratios(self, model, scale_spike_rasters):
        nb_total_units = np.sum([g.nb_units for g in model.groups[1:-1]])
        hr = (
            [1]
            + [
                scale_spike_rasters * g.nb_units / nb_total_units
                for g in model.groups[1:-1]
            ]
            + [1]
        )
        hr = list(reversed(hr))  # since we are plotting from bottom to top
        return hr

    def plot_input(
        self, ax, data, color, point_size, marker, point_alpha, set_axis=False
    ):
        ax.plot(data, color=color)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    def plot_readout(self, ax, data, label, pal, bg_col):
        ax.plot(label, color=bg_col)
        ax.plot(data, color=pal[0])
        self.turn_axis_off(ax)


def plot_training(
    results,
    nb_epochs,
    epoch_chunks=1,
    names=[
        "loss",
        "r2",
    ],
    save_path=None,
):
    fig, ax = plt.subplots(
        1,
        len(names),
        figsize=(2.5 * len(names), 2),
        dpi=150,
        sharex=True,
        constrained_layout=True,
    )

    for i, n in enumerate(names):

        ax[i].plot(results["{}".format(n)], color="black", label="train")
        ax[i].plot(results["val_{}".format(n)], color="black", alpha=0.5, label="valid")

        try:
            ax[i].scatter(
                [nb_epochs * (e + 1) for e in range(epoch_chunks)],
                results["test_{}".format(n)],
                color="coral",
                label="test",
            )
        except Exception as e:
            print(e)

        ax[i].set_ylabel(n)

        if "acc" in n:
            ax[i].set_ylim(0, 1)
        if "loss" in n:
            ax[i].set_yscale("log")
        if "r2" in n:
            ax[i].set_ylim(-0.01, 1.01)

        ax[i].set_xlabel("Epochs")

    ax[-1].legend()
    ax[0].set_xlabel("Epochs")

    sns.despine()

    if save_path is not None:
        fig.savefig(save_path, dpi=250)
    return fig, ax


def plot_precise_balance_example_neuron(
    results,
    key_exc="Pre test syne-Hid. 1 exc",
    key_inh="Pre test syni-Hid. 1 exc",
    idx_neuron=0,
    thr=None,
    scaling=True,
):
    """
    Plot a scatter plot of the excitatory and inhibitory synaptic currents for a single neuron at each time step for each stimulus
    """

    syne = results[key_exc][:, :, idx_neuron].flatten()
    syni = -results[key_inh][:, :, idx_neuron].flatten()

    if scaling:
        scl = np.max([np.max(syne), np.max(-syni)])
        syne = syne / scl
        syni = syni / scl

    if thr is not None:
        mask = (syne > thr) | (syni < -thr)

        syne = syne[mask]
        syni = syni[mask]

    lim = max(np.max(syne), np.max(-syni))

    slope, intercept, r_value, _, _ = linregress(syne, -syni)
    x = np.linspace(0, 100, 100)

    fig = plt.figure(figsize=(4, 4), dpi=150)
    plt.scatter(
        syne,
        -syni,
        s=1,
        color="black",
        alpha=0.1,
    )
    plt.plot([0, 100], [0, 100], color="silver")

    # plot linear fit
    plt.plot(
        x,
        slope * x + intercept,
        color="red",
        label=f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.2f}",
    )
    plt.legend()

    sns.despine()
    plt.xlabel("Excitatory synaptic current")
    plt.ylabel("Inhibitory synaptic current")
    plt.xlim(-0.01, lim)
    plt.ylim(-0.01, lim)
    plt.title(f"Precise balance - Neuron {idx_neuron}")

    return fig


def balance_idx(syne, syni, eps=1e-8):

    num = (syni + syne) ** 2
    denom = (syne - syni) ** 2 + eps
    return np.mean(num / denom)


def avg_precise_balance(
    results,
    key_exc="Pre test syne-Hid. 1 exc",
    key_inh="Pre test syni-Hid. 1 exc",
    thr=None,
    nb_exc_neurons=80,
    scaling=True,
):
    """
    Calculate the precise balance index averaged over all neurons.
    """
    precise_balance = []

    all_syne = results[key_exc]
    all_syni = -results[key_inh]

    if scaling:
        scl = np.max([np.max(all_syne), np.max(-all_syni)])
        all_syne = all_syne / scl
        all_syni = all_syni / scl

    print(balance_idx(all_syne.flatten(), all_syni.flatten()))

    for i in range(nb_exc_neurons):
        syne = all_syne[:, :, i].flatten()
        syni = all_syni[:, :, i].flatten()

        if thr is not None:

            mask = (syne > thr) | (syni < -thr)

            syne = syne[mask]
            syni = syni[mask]

        precise_balance.append(balance_idx(syne, syni))

    return np.mean(precise_balance)


# Detailed balance
def plot_detailed_balance_example_neuron(
    results,
    key_exc="Pre test syne-Hid. 1 exc",
    key_inh="Pre test syni-Hid. 1 exc",
    idx_neuron=0,
    scaling=True,
):
    """
    Plot a scatter plot of the mean excitatory and inhibitory synaptic currents for a single neuron.
    """

    mean_time_syne = results[key_exc][:, :, idx_neuron].mean(axis=1)
    mean_time_syni = -results[key_inh][:, :, idx_neuron].mean(axis=1)

    if scaling:
        scl = np.max([np.max(mean_time_syne), np.max(-mean_time_syni)])
        mean_time_syne = mean_time_syne / scl
        mean_time_syni = mean_time_syni / scl

    lim = max(np.max(mean_time_syne), np.max(-mean_time_syni)) * 1.1

    slope, intercept, r_value, p_value, std_err = linregress(
        mean_time_syne, -mean_time_syni
    )
    x = np.linspace(0, 100, 100)

    fig = plt.figure(figsize=(4, 4), dpi=150)
    plt.scatter(
        mean_time_syne,
        -mean_time_syni,
        s=1,
        color="black",
        alpha=0.5,
    )
    plt.plot([0, 100], [0, 100], color="silver")
    plt.plot(
        x,
        slope * x + intercept,
        color="red",
        label=f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.2f}",
    )
    plt.legend()

    sns.despine()
    plt.xlabel("Excitatory synaptic current")
    plt.ylabel("Inhibitory synaptic current")
    plt.title(f"Detailed balance - Neuron {idx_neuron}")

    plt.xlim(0, lim)
    plt.ylim(0, lim)

    return fig


def avg_detailed_balance(
    results,
    key_exc="Pre test syne-Hid. 1 exc",
    key_inh="Pre test syni-Hid. 1 exc",
    nb_exc_neurons=80,
    scaling=True,
):
    """
    Calculate the detailed balance index averaged over all neurons.
    """
    detailed_balance = []

    all_syne = results[key_exc]
    all_syni = -results[key_inh]

    if scaling:
        scl = np.max([np.max(all_syne), np.max(-all_syni)])
        all_syne = all_syne / scl
        all_syni = all_syni / scl

    for i in range(nb_exc_neurons):
        syne = all_syne[:, :, i].mean(axis=1)
        syni = all_syni[:, :, i].mean(axis=1)

        detailed_balance.append(balance_idx(syne, syni))

    return np.mean(detailed_balance)


# Tight balance
def plot_tight_balance_example_neuron(
    results,
    key_exc="Pre test syne-Hid. 1 exc",
    key_inh="Pre test syni-Hid. 1 exc",
    idx_neuron=0,
    scaling=True,
):
    """
    Plot a scatter plot of the mean excitatory and inhibitory synaptic currents for a single neuron.
    """

    mean_stimuli_syne = results[key_exc][:, :, idx_neuron].mean(axis=0)
    mean_stimuli_syni = -results[key_inh][:, :, idx_neuron].mean(axis=0)

    if scaling:
        scl = np.max([np.max(mean_stimuli_syne), np.max(-mean_stimuli_syni)])
        mean_stimuli_syne = mean_stimuli_syne / scl
        mean_stimuli_syni = mean_stimuli_syni / scl

    lim = max(np.max(mean_stimuli_syne), np.max(-mean_stimuli_syni)) * 1.1

    slope, intercept, r_value, p_value, std_err = linregress(
        mean_stimuli_syne, -mean_stimuli_syni
    )
    x = np.linspace(0, 100, 100)

    fig = plt.figure(figsize=(4, 4), dpi=150)
    plt.scatter(
        mean_stimuli_syne,
        -mean_stimuli_syni,
        s=1,
        color="black",
        alpha=np.arange(0.01, 1, 1 / len(mean_stimuli_syne)),
    )
    plt.plot([0, 100], [0, 100], color="silver")
    plt.plot(
        x,
        slope * x + intercept,
        color="red",
        label=f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.2f}",
    )
    plt.legend()

    sns.despine()
    plt.title(f"Tight balance - Neuron {idx_neuron}")
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel("Excitatory synaptic current")
    plt.ylabel("Inhibitory synaptic current")

    return fig


def avg_tight_balance(
    results,
    key_exc="Pre test syne-Hid. 1 exc",
    key_inh="Pre test syni-Hid. 1 exc",
    nb_exc_neurons=80,
    scaling=True,
):
    """
    Calculate the detailed balance index averaged over all neurons.
    """

    all_syne = results[key_exc]
    all_syni = -results[key_inh]

    if scaling:
        scl = np.max([np.max(all_syne), np.max(-all_syni)])
        all_syne = all_syne / scl
        all_syni = all_syni / scl

    tight_balance = []

    for i in range(nb_exc_neurons):
        syne = all_syne[:, :, i].mean(axis=0)
        syni = all_syni[:, :, i].mean(axis=0)

        tight_balance.append(balance_idx(syne, syni))

    return np.mean(tight_balance)


# global balance


def plot_global_balance(
    results,
    key_exc="Pre test syne-Hid. 1 exc",
    key_inh="Pre test syni-Hid. 1 exc",
    scaling=True,
):
    """
    Plot a scatter plot of the mean excitatory and inhibitory synaptic currents for all neurons.
    """

    # Calculate mean excitatory and inhibitory synaptic currents across all neurons
    mean_syne = np.mean(results[key_exc], axis=(0, 1))
    mean_syni = np.mean(-results[key_inh], axis=(0, 1))

    if scaling:
        scl = np.max([np.max(mean_syne), np.max(-mean_syni)])
        mean_syne = mean_syne / scl
        mean_syni = mean_syni / scl

    lim = max(np.max(mean_syne), np.max(-mean_syni)) * 1.1

    slope, intercept, r_value, p_value, std_err = linregress(mean_syne, -mean_syni)
    x = np.linspace(0, 100, 100)

    fig = plt.figure(figsize=(4, 4), dpi=150)
    plt.scatter(
        mean_syne,
        -mean_syni,
        s=1,
        color="black",
        alpha=1,
    )
    plt.plot([0, 100], [0, 100], color="silver")
    plt.plot(
        x,
        slope * x + intercept,
        color="red",
        label=f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.2f}",
    )
    plt.legend()

    sns.despine()
    plt.xlabel("Excitatory synaptic current")
    plt.ylabel("Inhibitory synaptic current")
    plt.title("Global balance")
    plt.xlim(0, lim)
    plt.ylim(0, lim)

    return fig


def avg_global_balance(
    results,
    key_exc="Pre test syne-Hid. 1 exc",
    key_inh="Pre test syni-Hid. 1 exc",
    scaling=True,
):
    """
    Calculate the global balance index.
    """
    all_syne = results[key_exc]
    all_syni = -results[key_inh]

    if scaling:
        scl = np.max([np.max(all_syne), np.max(-all_syni)])
        all_syne = all_syne / scl
        all_syni = all_syni / scl

    mean_syne = np.mean(all_syne, axis=(0, 1))
    mean_syni = np.mean(all_syni, axis=(0, 1))

    return balance_idx(mean_syne, mean_syni)


def plot_example_neuron_currents(
    results,
    key_exc="Pre test syne-Hid. 1 exc",
    key_inh="Pre test syni-Hid. 1 exc",
    idx_neuron=0,
):
    """
    Plot the synaptic currents of a single neuron.
    """
    syne = results[key_exc][:, :, idx_neuron]
    syni = -results[key_inh][:, :, idx_neuron]

    plt.figure(figsize=(4, 2.5), dpi=150)
    fig = plt.plot(syne.T, color="tab:red", alpha=0.01, label="Exc.")
    fig = plt.plot(syni.T, color="tab:blue", alpha=0.01, label="Inh.")
    fig = plt.plot(syne.T + syni.T, color="dimgray", alpha=0.01, label="Diff.")

    fig = plt.plot(syne.mean(axis=0), color="crimson", label="Exc.")
    fig = plt.plot(syni.mean(axis=0), color="navy", label="Inh.")
    fig = plt.plot(syne.mean(axis=0) + syni.mean(axis=0), color="black", label="Diff.")
    sns.despine()
    plt.xlabel("Time step")
    plt.ylabel("Synaptic current")
    plt.title(f"Neuron {idx_neuron}")

    return fig
