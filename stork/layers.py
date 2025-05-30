# Layers module
# Julian, October 2021

from . import connections
from . import nodes as nd
from . import utils
from . import constraints
from . import activations
from typing import Iterable


class AbstractLayer:
    """
    Abstract base class for layer object
    """

    def __init__(self, name, model, recurrent, dalian=False) -> None:
        super().__init__()

        self.model = model
        self.name = name
        self.recurrent = recurrent
        self.dalian = dalian

        # Lists that hold neurons and connections in this layer
        self.neurons = []
        self.connections = []

    def add_connection(self, connection, recurrent="detect", inhibitory=False):
        """
        Adds a connection to the layer

        :param source:              source group
        :param destination:         target group
        :param connection_class:    connection class to use
        :param recurrent:           whether connection is considered recurrent or not
        :param inhibitory:          whether the neurons are inhibitory or not (for dalian layers)
        """

        # Assert that the target neuron group is in the layer
        assert (
            connection.dst in self.neurons
        ), "Target neuron group is not in this Layer"

        # Detect whether connection is recurrent or not
        if recurrent == "detect":
            recurrent = True if connection.src == connection.dst else False
        else:
            assert isinstance(recurrent, bool)

        # Add is_recurrent and is_inhibitory boolean to connection
        # This is used by some initializer classes to calculate optimal weight initializations
        connection.is_recurrent = recurrent
        connection.is_inhibitory = inhibitory

        # Add pointers to the connection to the model, layer and neuron object
        self.model.add_connection(connection)
        connection.dst.afferents.append(connection)
        self.connections.append(connection)

    def add_neurons(self, neurons, inhibitory=False):
        """
        Adds a neuron group to the layer

        :param neurons:     neuron group
        """

        # Add afferents list to neuron group
        neurons.afferents = []

        # Flag inhibitory
        neurons.is_inhibitory = inhibitory

        # Add pointers to self and model
        self.neurons.append(neurons)
        self.model.add_group(neurons)


class SimpleLayer(AbstractLayer):
    """
    Implements a 'Layer' class that wraps multiple 'nodes' and 'connection' objects
    and adds them to an instance of an nn.Module.

    The 'Layer' class fulfills the following purpose:
        1.  Provide an easy-to-use and easy-to-modify constructor for each layer of a neural network
        2.  Enable layer-wise initialization strategies. Some initializer classes (in the 'initializers.py') module
            take a 'Layer' object as input and initialize all connections in the layer.

    The 'Layer' class is only a constructor and does NOT inherit from `nn.Module`, nor does it
    add a pointer to itself to the model. This could be differently implemented in the future
    """

    def __init__(
        self,
        name,
        model,
        size,
        input_group,
        recurrent=True,
        regs=None,
        w_regs=None,
        connection_class=connections.Connection,
        neuron_class=nd.LIFGroup,
        flatten_input_layer=True,
        neuron_kwargs={},
        connection_kwargs={},
    ) -> None:
        super().__init__(name, model, recurrent)

        # Make neuron group
        nodes = neuron_class(size, name=self.name, regularizers=regs, **neuron_kwargs)
        self.add_neurons(nodes)

        # Make afferent connection
        con = connection_class(
            input_group,
            nodes,
            regularizers=w_regs,
            flatten_input=flatten_input_layer,
            **connection_kwargs
        )
        self.add_connection(con)

        # Make recurrent connection
        if recurrent:
            con = connection_class(
                nodes, nodes, regularizers=w_regs, **connection_kwargs
            )
            self.add_connection(con)

        self.output_group = nodes


class ConvLayer(AbstractLayer):
    """
    Implements a spiking Convolutional Layer
    """

    def __init__(
        self,
        name,
        model,
        input_group,
        kernel_size,
        stride,
        padding=0,
        nb_filters=16,
        shape="auto",
        recurrent=True,
        regs=None,
        w_regs=None,
        connection_class=connections.ConvConnection,
        neuron_class=nd.LIFGroup,
        neuron_kwargs={},
        connection_kwargs={},
        recurrent_connection_kwargs={},
    ) -> None:
        super().__init__(name, model, recurrent)

        # Calculate size of Convolutional Layer
        # Must provide either the exact `shape` or a `nb_filters` parameter

        if shape == "auto":
            assert isinstance(
                nb_filters, int
            ), "Must provide nb_filters to calculate ConvLayer shape"

            shape = utils.convlayer_size(
                nb_inputs=input_group.shape[1:],
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )

            shape_dim = len(input_group.shape) - 1
            if shape_dim == 1:
                shape = tuple([nb_filters, int(shape[0])])
            else:
                shape = tuple([nb_filters] + [int(i) for i in shape])
        else:
            assert isinstance(
                shape, tuple
            ), "`shape` must be 'auto' or a tuple of integers"

        # Make neuron group
        nodes = neuron_class(shape, name=self.name, regularizers=regs, **neuron_kwargs)
        self.add_neurons(nodes)

        # Make afferent connection
        con = connection_class(
            input_group,
            nodes,
            regularizers=w_regs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **connection_kwargs
        )
        self.add_connection(con)

        # Make recurrent connection
        if recurrent:
            rec_kernel_size = recurrent_connection_kwargs.pop("kernel_size", 5)
            rec_stride = recurrent_connection_kwargs.pop("stride", 1)
            rec_padding = recurrent_connection_kwargs.pop("padding", 2)

            con = connection_class(
                nodes,
                nodes,
                regularizers=w_regs,
                kernel_size=rec_kernel_size,
                stride=rec_stride,
                padding=rec_padding,
                **recurrent_connection_kwargs
            )
            self.add_connection(con)

        self.output_group = nodes


class SimpleDalianLayer(AbstractLayer):
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
            size, name=self.name + " exc", regularizers=regs, **exc_neuron_kwargs
        )
        self.add_neurons(nodes_exc)

        # Make Inh neuron group
        nodes_inh = neuron_class(
            size_inh, name=self.name + " inh", regularizers=regs, **inh_neuron_kwargs
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


######################################################################################################
# TODO: Add complex layer and complex dalian layer (maybe structure into folder with base layer and files with additional layers)
######################################################################################################


class Layer(AbstractLayer):
    """
    Implements a 'Layer' class that wraps multiple 'nodes' and 'connection' objects
    and adds them to an instance of an nn.Module.

    The 'Layer' class fulfills the following purpose:
        1.  Provide an easy-to-use and easy-to-modify constructor for each layer of a neural network
        2.  Enable layer-wise initialization strategies. Some initializer classes (in the 'initializers.py') module
            take a 'Layer' object as input and initialize all connections in the layer.

    The 'Layer' class is only a constructor and does NOT inherit from `nn.Module`, nor does it
    add a pointer to itself to the model. This could be differently implemented in the future
    """

    def __init__(
        self,
        name,
        model,
        size,
        input_group,
        recurrent=True,
        regs=None,
        w_regs=None,
        connection_class=connections.Connection,
        neuron_class=nd.LIFGroup,
        flatten_input_layer=True,
        neuron_kwargs={},
        connection_kwargs={},
        ff_connection_kwargs={},
        rec_connection_kwargs={},
    ) -> None:
        super().__init__(name, model, recurrent)

        # Make neuron group
        nodes = neuron_class(size, name=self.name, regularizers=regs, **neuron_kwargs)
        self.add_neurons(nodes)

        # Make afferent connection
        if len(ff_connection_kwargs) == 0:
            ff_connection_kwargs = connection_kwargs
        if len(rec_connection_kwargs) == 0:
            rec_connection_kwargs = connection_kwargs

        con = connection_class(
            input_group,
            nodes,
            regularizers=w_regs,
            flatten_input=flatten_input_layer,
            **ff_connection_kwargs
        )
        self.add_connection(con)

        # Make recurrent connection
        if recurrent:
            con = connection_class(
                src=nodes, dst=nodes, regularizers=w_regs, **rec_connection_kwargs
            )
            self.add_connection(con)

        self.output_group = nodes


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
        mask=None,
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

        # cut mask in parts for each connection
        if mask is not None:
            if isinstance(size, Iterable):
                raise NotImplementedError(
                    "Masking is not implemented for Convolutional layers"
                )
            elif mask.shape[0] != size + size_inh:
                raise ValueError("Mask shape does not match the size of the layer")
            else:
                print("." * 100)
                print("using mask")
                print(size, size_inh)
                print(mask.shape)
                mask_ee = mask[:size, :size]
                mask_ie = mask[:size, size:]
                mask_ei = mask[size:, :size]
                mask_ii = mask[size:, size:]

                print(mask_ee.shape, mask_ei.shape, mask_ie.shape, mask_ii.shape)

                rec_exc_connection_kwargs["src_blocks"] = None
                rec_exc_connection_kwargs["dst_blocks"] = None
                rec_inh_connection_kwargs["src_blocks"] = None
                rec_inh_connection_kwargs["dst_blocks"] = None

                print("mask_ee", mask_ee.requires_grad)
                print("." * 100)

        print(rec_exc_connection_kwargs)

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
        if "exc" in ff_connection_kwargs:
            exc_ff_connection_kwargs = ff_connection_kwargs["exc"]
            inh_ff_connection_kwargs = ff_connection_kwargs["inh"]
        else:
            exc_ff_connection_kwargs = inh_ff_connection_kwargs = ff_connection_kwargs

        print(
            # "Careful: Input is added directly to the membrane of the neurons (for feedforward input). "
            "Careful: need to change target if you want to add input directly to the membrane"
        )
        print(connection_class)
        con_XE = connection_class(
            input_group,
            nodes_exc,
            name="XE",
            target="exc",
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
            target="exc",
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

        if "src_blocks" in rec_exc_connection_kwargs:
            rec_inh_connection_kwargs["dst_blocks"] = rec_exc_connection_kwargs[
                "src_blocks"
            ]

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

            if "src_blocks" in rec_inh_connection_kwargs:
                rec_exc_connection_kwargs["dst_blocks"] = rec_inh_connection_kwargs[
                    "src_blocks"
                ]

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
