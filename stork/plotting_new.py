import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stork import datasets


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
    def __init__(self):
        super().__init__()

    def get_activities(self, model, data):

        # Run model once and get activities
        scores = model.evaluate(data, one_batch=True).tolist()

        inp = model.input_group.get_flattened_out_sequence().detach().cpu().numpy()
        hidden_groups = model.groups[1:-1]
        hid_activity = [
            g.get_flattened_out_sequence().detach().cpu().numpy() for g in hidden_groups
        ]
        out_group = model.out.detach().cpu().numpy()

        return inp, hid_activity, out_group

    def get_height_ratios(self, model, scale_spike_rasters):
        nb_total_units = np.sum([g.nb_units for g in model.groups[:-1]])
        hr = [
            scale_spike_rasters * g.nb_units / nb_total_units for g in model.groups[:-1]
        ] + [1]
        hr = list(reversed(hr))  # since we are plotting from bottom to top
        return hr

    def plot_input(
        self, ax, data, color, point_size, marker, point_alpha, set_axis=False
    ):
        self.plot_spike_raster(
            ax,
            data,
            color,
            point_size,
            marker,
            point_alpha,
            set_axis,
        )

    def plot_hidden(
        self, ax, data, color, point_size, marker, point_alpha, set_axis=False
    ):
        self.plot_spike_raster(
            ax,
            data,
            color,
            point_size,
            marker,
            point_alpha,
            set_axis,
        )

    def plot_readout(self, ax, data, label, pal, bg_col):

        for line_index, ro_line in enumerate(np.transpose(data)):
            if line_index == label:
                ax.plot(ro_line, color=pal[line_index])
            else:
                ax.plot(ro_line, color=bg_col, zorder=-5, alpha=0.5)

        self.turn_axis_off(ax)

    def plot_activity(
        self,
        model,
        data,
        nb_samples=2,
        samples=None,
        figsize=(5, 5),
        dpi=250,
        marker=".",
        point_size=5,
        point_alpha=1,
        pal=sns.color_palette("muted", n_colors=20),
        bg_col="#AAAAAA",
        scale_spike_rasters=4,
        color_input=True,
    ):
        print("plotting")
        inp, hid_activity, out_group = self.get_activities(model, data)
        nb_groups = len(hid_activity)

        hr = self.get_height_ratios(model, scale_spike_rasters)

        fig, ax = plt.subplots(
            nb_groups + 2,
            nb_samples,
            figsize=figsize,
            dpi=dpi,
            sharex="row",
            sharey="row",
            gridspec_kw={"height_ratios": hr},
        )

        sns.despine()

        if samples is None:
            samples = list(range(nb_samples))

        for i, s in enumerate(samples):

            # plot and color input spikes
            if color_input:
                c = pal[data[i][1]]
            else:
                c = bg_col
            self.plot_input(
                ax[-1, i],
                inp[s],
                c,
                point_size,
                marker,
                point_alpha,
                set_axis=True,
            )

            # plot hidden layer spikes
            for g in range(nb_groups):
                self.plot_hidden(
                    ax[-(2 + g), i],
                    hid_activity[g][s],
                    "k",
                    point_size / 2,
                    marker,
                    point_alpha,
                )
                ax[-(2 + g)][0].set_ylabel(model.groups[g + 1].name)

                if i == 0:
                    ax[-(2 + g)][i].set_yticks([])

            # plot readout
            self.plot_readout(ax[0][i], out_group[i], data[i][1], pal, bg_col)

            ax[-1][i].set_xlabel("Time (s)")

        ax[-1][0].set_ylabel("Input")
        ax[-1][0].set_yticks([])
        ax[-1][0].set_xlim(-3, model.nb_time_steps + 3)

        ax[0][0].set_ylabel("Readout")
        ax[0][0].set_yticks([])

        duration = round(model.nb_time_steps * model.time_step * 10) / 10
        ax[-1][0].set_xticks([0, model.nb_time_steps], [0, duration])

        plt.tight_layout()


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
