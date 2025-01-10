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
        readout_layers = [-1],
        batch_size=None
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

        self.nb_samples = nb_samples
        self.samples = samples
        self.figsize = figsize
        self.dpi = dpi
        self.pal = pal
        self.bg_col = bg_col

        self.model = model
        self.data = data
        self.readout_layers = readout_layers

    def get_activities(self):

        # Run model once and get activities
        scores = self.model.evaluate(self.data, two_batches=True).tolist()
        activities = [g.get_flattened_out_sequence().detach().cpu().numpy() for g in self.model.groups]

        if self.batch_size is None:
            self.batch_size = len(activities[0])

        return activities

    def get_height_ratios(self):
        nb_groups = len(self.model.groups)
        nb_spiking_groups = nb_groups -len(self.readout_layers)

        nb_total_units = np.sum([self.model.groups[g].nb_units for g in range(nb_spiking_groups)])
        hr = [
            self.scale_spike_rasters * self.model.groups[g].nb_units / nb_total_units
            for g in range(nb_spiking_groups)
        ] + [1]*len(self.readout_layers)
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
            ax.plot(label, color=self.label_color)

        self.turn_axis_off(ax)

    def plot_activity(
        self,
    ):
        print("plotting")
        activities = self.get_activities()
        nb_groups = len(self.model.groups)
        nb_spiking_groups = nb_groups -len(self.readout_layers)

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

        sns.despine()

        if self.samples is None:
            self.samples = list(range(self.nb_samples))


        ##############################################################################################
        # Plotting
        ##############################################################################################

        ylims = {g: (float('inf'), float('-inf')) for g in range(nb_groups)}
        for i, s in enumerate(self.samples):

            self.tuple_label = False

            # plot and color input spikes
            if self.color_classes:
                try:
                    c = self.pal[self.data[s+self.batch_size][1]]
                except:
                    c = self.pal[self.data[s+self.batch_size][1][1]]
                    self.tuple_label = True
            else:
                c = self.bg_col



            self.plot_input(
                ax[-1, i],
                activities[0][s],
                c,
            )
            ax[-1][0].set_ylabel(self.model.groups[0].name)


            # plot hidden layer spikes
            for g in range(1, nb_spiking_groups):
                self.plot_hidden(
                    ax[-(1 + g), i],
                    activities[g][s],
                )
                ax[-(1 + g)][0].set_ylabel(self.model.groups[g].name)

                if i == 0:
                    ax[-(1 + g)][i].set_yticks([])

            # plot readout
            for g in range(nb_spiking_groups, nb_groups):
                self.plot_readout(
                    ax[-(1 + g), i], activities[g][s], self.data[s+self.batch_size][1], self.pal, self.bg_col
                )
                ax[-(1 + g), 0].set_ylabel(self.model.groups[g].name)
                ax[-(1 + g)][0].set_yticks([])

                ylims[g] = (min(ylims[g][0], min(activities[g][s].flatten())), max(ylims[g][1], max(activities[g][s].flatten())))


            ax[-1][i].set_xlabel("Time (s)")


        for g in range(nb_spiking_groups,nb_groups):
            ax[-(1 + g)][0].set_ylim(ylims[g])





        ax[-1][0].set_ylabel("Input")
        ax[-1][0].set_yticks([])
        ax[-1][0].set_xlim(-3, self.model.nb_time_steps + 3)


        duration = round(self.model.nb_time_steps * self.model.time_step * 10) / 10
        ax[-1][0].set_xticks([0, self.model.nb_time_steps], [0, duration])


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
