import tensorflow as tf
import keras.backend as K
import keras.layers as layers
import keras.initializers as kinit
import keras.regularizers as regularizers
import keras.constraints as constraints
import keras.models 
import sys 
import numpy as np 

thismodule = sys.modules[__name__]

def main():

    spec = [
        {
            "name" : "topology",
            "hidden_activation" : "tanh",
            "layer_sizes" : [5],
            "type" : "ff",
        },
        {
            "name" : "go",
            "hidden_activation" : "tanh",
            "layer_sizes" : [5],
            "type" : "ff",
        },
        {
            "name" : "localization",
            "hidden_activation" : "tanh",
            "kernel_size" : 5,
            "locations" : 2,
            "type" : "loc",
            "output_size" : 5,
        }
    ]

    arch = create_input_architecture(output_size=5, output_activation='tanh', 
        name='single_input', spec=spec)

    inputs_a = create_input_nodes(spec, [(25,),(150,),(2,9)], base_name="a")
    inputs_b = create_input_nodes(spec,[(25,),(150,),(2,9)],  base_name="b")
    
    output_a = arch(inputs_a, name='input_a')
    output_b = arch(inputs_b, name='input_b')

    inputs = inputs_a + inputs_b

    model = keras.models.Model(inputs=inputs, outputs=[output_a , output_b])
    print(model.summary())
    keras.utils.plot_model(model, '../tmp/model.png', show_shapes=True)

def create_input_nodes(spec, shapes, base_name=""):
    """ creates the input nodes, given a spec """
    input_nodes = []
    for i, elm in enumerate(spec):
        input_node = layers.Input(shape=shapes[i], name="%s_%s" % (elm['name'], base_name))
        print("Created input %s with shape %s" % (input_node.name, str(shapes[i])))
        input_nodes.append(input_node)
    
    return input_nodes


def compartment_ff(layer_sizes, hidden_activation, name, **kwargs):
    """ simple feed-forward fully connected compartment """ 
    
    comp_layers = []
    for i, hidden_layer_size in enumerate(layer_sizes):
        layer = layers.Dense(hidden_layer_size, 
                             activation=hidden_activation,
                             name="ff_%d_%s" % (i, name))
        comp_layers.append(layer)
    
    def apply(input_node):
        layer = input_node
        for l in comp_layers:
            layer = l(layer)
        return layer 

    return apply 

def compartment_loc(kernel_size, output_size, hidden_activation, name, locations, **kwargs):
    """ 
        specialized compartment to handle localization features 
        a simple nn kernel is applied to the time series for each location and
        the outputs are passed through one final layer
    """
    nn = layers.Dense(kernel_size, activation=hidden_activation, name="loc_nn_%s" % name)
    output_nn = layers.Dense(output_size, activation=hidden_activation, name="loc_output_%s" % name)

    def apply(input_node):
        comp_outputs = []
        input_name = friendly_name(input_node)
        for loc in range(locations):
            row_slice = layers.Lambda(lambda x: x[:, loc, :], name="loc_slice_%s_%d" % (input_name, loc))(input_node)
            output = nn(row_slice)
            comp_outputs.append(output)
        output = concatenate(comp_outputs, name='loc_preoutput_%s' % input_name)
        output = output_nn(output)

        return output 
    
    return apply 

def create_input_architecture(output_size, output_activation, name, spec):
    """
        creates compartmentalized input architecture to a neural network
    """
    compartments = []
    for i, compartment_spec in enumerate(spec):
        func = getattr(thismodule, "compartment_%s" % compartment_spec['type'])
        applicator = func(**compartment_spec)
        compartments.append(applicator)
    
    output_nn = layers.Dense(output_size, activation=output_activation, name='arch_output_%s' % name)

    def apply(input_nodes, name):
        output_nodes = []
        for i, input_node in enumerate(input_nodes):
            output_node = compartments[i](input_node)
            output_nodes.append(output_node)

        merged = concatenate(output_nodes, name='arch_preoutput_%s' % name)
        merged = output_nn(merged)

        return merged

    return apply 

def concatenate(arr, **kwargs):
    if len(arr) == 1:
        return arr[0]
    
    return layers.Concatenate(**kwargs)(arr)

def friendly_name(l):
    return l.name.split(':')[0]

if __name__ == "__main__":
    main()
