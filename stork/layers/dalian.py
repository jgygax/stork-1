from .. import connections
from .. import nodes as nd
from .. import utils
from .. import constraints
from .. import activations
from typing import Iterable

from .base import AbstractLayer


class DalianLayer(AbstractLayer):
    """
    Implements a fully connected layer following Dale's law.
    Consists of one Excitatory and one Inhibitory population
    """

    def __init__(
        self,
        name,
        model,
        size,
        input_group,
        ei_ratio=4,
        recurrent=True,
        regs=None,
        w_regs=None,
        connection_class=connections.Connection,
        neuron_class=nd.ExcInhLIFGroup,
        flatten_input_layer=True,
        exc_neuron_kwargs={},
        inh_neuron_kwargs={},
        ff_connection_kwargs={},
        rec_inh_connection_kwargs={},
        rec_exc_connection_kwargs={},
        activation=activations.SuperSpike,
    ) -> None:
        super().__init__(name, model, recurrent=recurrent, dalian=True)

        # Add Dalian constraint
        pos_constraint = constraints.MinMaxConstraint(min=0.0)

        # Compute inhibitory layer size
        if isinstance(size, Iterable):
            # For conv layer
            size_inh = (int(tuple(size)[0] / ei_ratio),) + tuple(size)[1:]
        else:
            # For normal layer
            size_inh = int(size / ei_ratio)

        size = tuple(size) if isinstance(size, Iterable) else size

        # Make Exc neuron group
        nodes_exc = neuron_class(
            size,
            name=self.name + " exc",
            regularizers=regs,
            activation=activation,
            **exc_neuron_kwargs
        )
        self.add_neurons(nodes_exc)

        # Make Inh neuron group
        nodes_inh = neuron_class(
            size_inh,
            name=self.name + " inh",
            regularizers=regs,
            activation=activation,
            **inh_neuron_kwargs
        )
        self.add_neurons(nodes_inh, inhibitory=True)

        # Make afferent connections
        con_XE = connection_class(
            input_group,
            nodes_exc,
            name="XE",
            regularizers=w_regs,
            constraints=pos_constraint,
            **ff_connection_kwargs,
            flatten_input=flatten_input_layer
        )
        self.add_connection(con_XE, recurrent=False, inhibitory=False)

        con_XI = connection_class(
            input_group,
            nodes_inh,
            name="XI",
            regularizers=w_regs,
            constraints=pos_constraint,
            **ff_connection_kwargs,
            flatten_input=flatten_input_layer
        )
        self.add_connection(con_XI, recurrent=False, inhibitory=False)

        # RECURRENT CONNECTIONS: INHIBITORY
        # # # # # # # # # # # # # #

        con_II = connection_class(
            nodes_inh,
            nodes_inh,
            target="inh",
            name="II",
            regularizers=w_regs,
            constraints=pos_constraint,
            **rec_inh_connection_kwargs
        )
        self.add_connection(con_II, recurrent=False, inhibitory=True)

        con_IE = connection_class(
            nodes_inh,
            nodes_exc,
            target="inh",
            name="IE",
            regularizers=w_regs,
            constraints=pos_constraint,
            **rec_inh_connection_kwargs
        )
        self.add_connection(con_IE, recurrent=False, inhibitory=True)

        # RECURRENT CONNECTIONS: EXCITATORY
        # # # # # # # # # # # # # #

        if recurrent:
            con_EI = connection_class(
                nodes_exc,
                nodes_inh,
                name="EI",
                regularizers=w_regs,
                constraints=pos_constraint,
                **rec_exc_connection_kwargs
            )
            self.add_connection(con_EI, recurrent=True, inhibitory=False)

            con_EE = connection_class(
                nodes_exc,
                nodes_exc,
                name="EE",
                regularizers=w_regs,
                constraints=pos_constraint,
                **rec_exc_connection_kwargs
            )
            self.add_connection(con_EE, recurrent=True, inhibitory=False)

        self.output_group = nodes_exc


class StructuredDalianLayer(AbstractLayer):
    """
    Implements a fully connected layer following Dale's law.
    Consists of one Excitatory and one Inhibitory population
    """

    def __init__(
        self,
        name,
        model,
        size,
        input_group,
        ei_ratio=4,
        recurrent=True,
        regs=None,
        w_regs=None,
        connection_class=connections.Connection,
        neuron_class=nd.ExcInhLIFGroup,
        flatten_input_layer=True,
        exc_neuron_kwargs={},
        inh_neuron_kwargs={},
        ff_connection_kwargs={},
        rec_connection_kwargs={},
        activation=activations.SuperSpike,
        ff_target="exc",
        mask=None,
    ) -> None:
        super().__init__(name, model, recurrent=recurrent, dalian=True)

        # Add Dalian constraint
        pos_constraint = constraints.MinMaxConstraint(min=0.0)

        # Update kwargs
        if "exc" in ff_connection_kwargs:
            exc_ff_connection_kwargs = ff_connection_kwargs["exc"]
            inh_ff_connection_kwargs = ff_connection_kwargs["inh"]
        else:
            exc_ff_connection_kwargs = inh_ff_connection_kwargs = ff_connection_kwargs
        if "exc" in rec_connection_kwargs:
            rec_exc_connection_kwargs = rec_connection_kwargs["exc"]
            rec_inh_connection_kwargs = rec_connection_kwargs["inh"]
        else:
            rec_exc_connection_kwargs = rec_inh_connection_kwargs = (
                rec_connection_kwargs
            )

        # cut mask in parts for each connection
        if mask is not None:
            if isinstance(size, Iterable):
                raise NotImplementedError(
                    "Masking is not implemented for Convolutional layers"
                )
            else:
                mask_ee = mask[:size, :size]
                mask_ie = mask[:size, size:]
                mask_ei = mask[size:, :size]
                mask_ii = mask[size:, size:]

        # Compute inhibitory layer size
        if isinstance(size, Iterable):
            # For conv layer
            size_inh = (int(tuple(size)[0] / ei_ratio),) + tuple(size)[1:]
        else:
            # For normal layer
            size_inh = int(size / ei_ratio)

        size = tuple(size) if isinstance(size, Iterable) else size

        # Make Exc neuron group
        nodes_exc = neuron_class(
            size,
            name=self.name + " exc",
            regularizers=regs,
            activation=activation,
            **exc_neuron_kwargs
        )
        self.add_neurons(nodes_exc)

        # Make Inh neuron group
        nodes_inh = neuron_class(
            size_inh,
            name=self.name + " inh",
            regularizers=regs,
            activation=activation,
            **inh_neuron_kwargs
        )
        self.add_neurons(nodes_inh, inhibitory=True)

        # Make afferent connections
        con_XE = connection_class(
            input_group,
            nodes_exc,
            name="XE",
            target=ff_target,
            regularizers=w_regs,
            constraints=pos_constraint,
            **exc_ff_connection_kwargs,
            flatten_input=flatten_input_layer
        )
        self.add_connection(con_XE, recurrent=False, inhibitory=False)

        con_XI = connection_class(
            input_group,
            nodes_inh,
            name="XI",
            target=ff_target,
            regularizers=w_regs,
            constraints=pos_constraint,
            **inh_ff_connection_kwargs,
            flatten_input=flatten_input_layer
        )
        self.add_connection(con_XI, recurrent=False, inhibitory=False)

        # RECURRENT CONNECTIONS: INHIBITORY
        # # # # # # # # # # # # # #

        if mask is not None:
            rec_inh_connection_kwargs["mask"] = mask_ii

        con_II = connection_class(
            nodes_inh,
            nodes_inh,
            target="inh",
            name="II",
            regularizers=w_regs,
            constraints=pos_constraint,
            **rec_inh_connection_kwargs
        )
        self.add_connection(con_II, recurrent=False, inhibitory=True)

        if mask is not None:
            rec_inh_connection_kwargs["mask"] = mask_ie

        con_IE = connection_class(
            nodes_inh,
            nodes_exc,
            target="inh",
            name="IE",
            regularizers=w_regs,
            constraints=pos_constraint,
            **rec_inh_connection_kwargs
        )
        self.add_connection(con_IE, recurrent=False, inhibitory=True)

        # RECURRENT CONNECTIONS: EXCITATORY
        # # # # # # # # # # # # # #

        if recurrent:

            if mask is not None:
                rec_exc_connection_kwargs["mask"] = mask_ee

            con_EE = connection_class(
                nodes_exc,
                nodes_exc,
                name="EE",
                regularizers=w_regs,
                constraints=pos_constraint,
                **rec_exc_connection_kwargs
            )
            self.add_connection(con_EE, recurrent=True, inhibitory=False)

            if mask is not None:
                rec_exc_connection_kwargs["mask"] = mask_ei

            con_EI = connection_class(
                nodes_exc,
                nodes_inh,
                name="EI",
                regularizers=w_regs,
                constraints=pos_constraint,
                **rec_exc_connection_kwargs
            )
            self.add_connection(con_EI, recurrent=True, inhibitory=False)

        self.output_group = nodes_exc


class TwoInhDalianLayer(AbstractLayer):
    """
    Implements a fully connected layer following Dale's law.
    Consists of one Excitatory and one Inhibitory population
    """

    def __init__(
        self,
        name,
        model,
        size,
        input_group,
        ei_ratio=4,
        recurrent=True,
        regs_exc=None,
        regs_inh=None,
        regs_inh2=None,
        w_regs=None,
        connection_class=connections.Connection,
        neuron_class=nd.ExcInhLIFGroup,
        neuron_class2=nd.ExcInhLIFGroup,
        flatten_input_layer=True,
        exc_neuron_kwargs={},
        inh_neuron_kwargs={},
        ff_connection_kwargs={},
        rec_inh_connection_kwargs={},
        rec_exc_connection_kwargs={},
        activation=activations.SuperSpike,
    ) -> None:
        super().__init__(name, model, recurrent=recurrent, dalian=True)

        # Add Dalian constraint
        pos_constraint = constraints.MinMaxConstraint(min=0.0)

        # Compute inhibitory layer size
        if isinstance(size, Iterable):
            # For conv layer
            size_inh = (int(tuple(size)[0] / ei_ratio),) + tuple(size)[1:]
        else:
            # For normal layer
            size_inh = int(size / ei_ratio)

        size = tuple(size) if isinstance(size, Iterable) else size

        # Make Exc neuron group
        nodes_exc = neuron_class(
            size,
            name=self.name + " exc",
            regularizers=regs_exc,
            activation=activation,
            **exc_neuron_kwargs
        )
        self.add_neurons(nodes_exc)

        # Make Inh neuron group
        nodes_inh = neuron_class(
            size_inh,
            name=self.name + " inh",
            regularizers=regs_inh,
            activation=activation,
            **inh_neuron_kwargs
        )
        self.add_neurons(nodes_inh, inhibitory=True)

        # Make 2nd Inh neuron group
        nodes_inh2 = neuron_class2(
            size_inh,
            name=self.name + " inh2",
            regularizers=regs_inh2,
            activation=activation,
            **inh_neuron_kwargs
        )
        self.add_neurons(nodes_inh2, inhibitory=True)

        # Make afferent connections
        con_XE = connection_class(
            input_group,
            nodes_exc,
            name="XE",
            regularizers=w_regs,
            constraints=pos_constraint,
            **ff_connection_kwargs,
            flatten_input=flatten_input_layer
        )
        self.add_connection(con_XE, recurrent=False, inhibitory=False)

        con_XI = connection_class(
            input_group,
            nodes_inh,
            name="XI",
            regularizers=w_regs,
            constraints=pos_constraint,
            **ff_connection_kwargs,
            flatten_input=flatten_input_layer
        )
        self.add_connection(con_XI, recurrent=False, inhibitory=False)

        con_XI2 = connection_class(
            input_group,
            nodes_inh2,
            name="XI2",
            regularizers=w_regs,
            constraints=pos_constraint,
            **ff_connection_kwargs,
            flatten_input=flatten_input_layer
        )
        self.add_connection(con_XI2, recurrent=False, inhibitory=False)

        # RECURRENT CONNECTIONS: INHIBITORY
        # # # # # # # # # # # # # #

        con_II = connection_class(
            nodes_inh,
            nodes_inh,
            target="inh",
            name="II",
            regularizers=w_regs,
            constraints=pos_constraint,
            **rec_inh_connection_kwargs
        )
        self.add_connection(con_II, recurrent=False, inhibitory=True)

        con_II2 = connection_class(
            nodes_inh2,
            nodes_inh2,
            target="inh",
            name="II2",
            regularizers=w_regs,
            constraints=pos_constraint,
            **rec_inh_connection_kwargs
        )
        self.add_connection(con_II2, recurrent=False, inhibitory=True)

        con_IE = connection_class(
            nodes_inh,
            nodes_exc,
            target="inh",
            name="IE",
            regularizers=w_regs,
            constraints=pos_constraint,
            **rec_inh_connection_kwargs
        )
        self.add_connection(con_IE, recurrent=False, inhibitory=True)

        con_I2E = connection_class(
            nodes_inh2,
            nodes_exc,
            target="inh",
            name="I2E",
            regularizers=w_regs,
            constraints=pos_constraint,
            **rec_inh_connection_kwargs
        )
        self.add_connection(con_I2E, recurrent=False, inhibitory=True)

        # RECURRENT CONNECTIONS: EXCITATORY
        # # # # # # # # # # # # # #

        if recurrent:
            con_EE = connection_class(
                nodes_exc,
                nodes_exc,
                name="EE",
                regularizers=w_regs,
                constraints=pos_constraint,
                **rec_exc_connection_kwargs
            )
            self.add_connection(con_EE, recurrent=True, inhibitory=False)

            if "src_blocks" in rec_inh_connection_kwargs:
                rec_exc_connection_kwargs["dst_blocks"] = rec_inh_connection_kwargs[
                    "src_blocks"
                ]

            con_EI = connection_class(
                nodes_exc,
                nodes_inh,
                name="EI",
                regularizers=w_regs,
                constraints=pos_constraint,
                **rec_exc_connection_kwargs
            )
            self.add_connection(con_EI, recurrent=True, inhibitory=False)

            con_EI2 = connection_class(
                nodes_exc,
                nodes_inh2,
                name="EI2",
                regularizers=w_regs,
                constraints=pos_constraint,
                **rec_exc_connection_kwargs
            )
            self.add_connection(con_EI2, recurrent=True, inhibitory=False)

        self.output_group = nodes_exc
