import numpy as np
import json


prefixes = ['softmax', 'fc', 'conv', 'max_pool', 'avg_pool', 'relu']  # TODO: ADD CONCAT

# Validate that every dictionary key is the name of a valid layer format
def validate_prefixes(names):
    for name in names:
        index = name.rfind('/')
        if index != -1: section = name[index + 1:]
        else: section = name
        hasPrefix = False
        for prefix in prefixes:
            if (section.startswith(prefix)):
                hasPrefix = True
                break

        if not hasPrefix:
            return False

    return True

# Prefix of the namespaces in the dictionary
prefix = '/home/peter/Desktop/'

# Max pool must have entry in dict mapped to a list of the format [windowHeight, windowWidth, strideHeight, strideWidth]
    # Also must be named 'max_pool' + etc.
# Average pool must have entry in dict mapped to a list of the format [windowHeight, windowWidth, strideHeight, strideWidth].
    # Also must be named 'avg_pool' + etc.
# Softmax must be named 'softmax' + etc.
# Full connected must be named 'fc' + etc.
# Convolutional layer must have entry in dict mapped to a list of the format [strideHeight, strideWidth, padding]
    # Padding is an optional entry for if you want custom padding, not 'SAME' padding,
def convert_separate(graph, namescopes, dict, session, channels, height, width):
    if not validate_prefixes(namescopes):
        return None
    # Create a model specification file named "input" that specifies input tensor parameters
    json_object = {}
    json_object['num_input_channels'] = channels
    json_object['input_height'] = height
    json_object['input_width'] = width
    with open(prefix + 'input', 'w') as outfile:
        json.dump(json_object, outfile)
        outfile.close()
    counter = 0

    # Create a model specification file for each layer in the network
    for namescope in namescopes:
        counter += 1
        index = namescope.rfind('/')
        if index != -1: section = namescope[index + 1:]
        else: section = namescope
        print(section)
        layer = {}
        if section.startswith(prefixes[0]):
            # If layer is softmax
            layer['name'] = 'softmax'
        elif section.startswith(prefixes[1]) and namescope not in dict:
            # If layer is fully connected
            layer['name'] = 'fc'
            for variable in graph.get_collection('trainable_variables', namescope):
                name = variable.name[len(namescope) + 1:]
                if name.startswith('weight'):
                    weight = session.run(variable)
                    layer['weights'] = weight.tolist()
                if name.startswith('bias'):
                    bias = session.run(variable)
                    layer['biases'] = bias.tolist()
                    layer['num_outputs'] = len(bias)
        elif section.startswith(prefixes[2]) or (namescope in dict and (len(dict[namescope]) == 2 or len(dict[namescope]) == 3)):
            # If layer is convolutional
            layer['name'] = 'conv'
            for variable in graph.get_collection('trainable_variables', namescope):
                name = variable.name[len(namescope) + 1:]
                if name.startswith('weight'):
                    weight = session.run(variable)
                    shape = weight.shape
                    layer['weights_hwio'] = np.transpose(weight, (3,2,0,1)).tolist()   # Rearrange order to be compatible with TensorRT
                    layer['filter_height'] = shape[0]
                    layer['filter_width'] = shape[1]
                    layer['out_maps'] = shape[3]
                if name.startswith('bias'):
                    bias = session.run(variable)
                    layer['biases'] = bias.tolist()
                    layer['num_outputs'] = len(bias)
            properties = dict[namescope]
            layer['stride_height'] = properties[0]
            layer['stride_width'] = properties[1]
            if (len(properties) == 3): layer['padding'] = properties[2]
            else: layer['padding'] = -1
            print(layer['padding'])
        elif section.startswith(prefixes[3]):
            # If layer is max pool
            layer['name'] = 'max_pool'
            properties = dict[namescope]
            layer['window_height'] = properties[0]
            layer['window_width'] = properties[1]
            layer['stride_height'] = properties[2]
            layer['stride_width'] = properties[3]
        elif section.startswith(prefixes[4]):
            # If layer is average pool
            layer['name'] = 'avg_pool'
            properties = dict[namescope]
            layer['window_height'] = properties[0]
            layer['window_width'] = properties[1]
            layer['stride_height'] = properties[2]
            layer['stride_width'] = properties[3]
        elif section.startswith(prefixes[5]):
            # If layer is a ReLU activation
            layer['name'] = 'relu'

        with open(prefix + str(counter), 'w') as outfile:
            json.dump(layer, outfile)
            outfile.close()


def convert_entire(graph, namescopes, dict, session, channels, height, width):
    if not validate_prefixes(namescopes):
        return None
    # Create a model specification file named "input" that specifies input tensor parameters
    json_object = {}
    json_object['num_input_channels'] = channels
    json_object['input_height'] = height
    json_object['input_width'] = width
    json_object['layers'] = []

    # Create a model specification file for each layer in the network
    for namescope in namescopes:
        index = namescope.rfind('/')
        if index != -1: section = namescope[index + 1:]
        else: section = namescope
        print(section)
        layer = {}
        if section.startswith(prefixes[0]):
            # If layer is softmax
            layer['name'] = 'softmax'
        elif section.startswith(prefixes[1]) and namescope not in dict:
            # If layer is fully connected
            layer['name'] = 'fc'
            for variable in graph.get_collection('trainable_variables', namescope):
                name = variable.name[len(namescope) + 1:]
                if name.startswith('weight'):
                    weight = session.run(variable)
                    layer['weights'] = weight.tolist()
                if name.startswith('bias'):
                    bias = session.run(variable)
                    layer['biases'] = bias.tolist()
                    layer['num_outputs'] = len(bias)
        elif section.startswith(prefixes[2]) or (namescope in dict and (len(dict[namescope]) == 2 or len(dict[namescope]) == 3)):
            # If layer is convolutional
            layer['name'] = 'conv'
            for variable in graph.get_collection('trainable_variables', namescope):
                name = variable.name[len(namescope) + 1:]
                if name.startswith('weight'):
                    weight = session.run(variable)
                    shape = weight.shape
                    layer['weights_hwio'] = np.transpose(weight, (3,2,0,1)).tolist()   # Rearrange order to be compatible with TensorRT
                    layer['filter_height'] = shape[0]
                    layer['filter_width'] = shape[1]
                    layer['out_maps'] = shape[3]
                if name.startswith('bias'):
                    bias = session.run(variable)
                    layer['biases'] = bias.tolist()
                    layer['num_outputs'] = len(bias)
            properties = dict[namescope]
            layer['stride_height'] = properties[0]
            layer['stride_width'] = properties[1]
            if (len(properties) == 3): layer['padding'] = properties[2]
            else: layer['padding'] = -1
            print(layer['padding'])
        elif section.startswith(prefixes[3]):
            # If layer is max pool
            layer['name'] = 'max_pool'
            properties = dict[namescope]
            layer['window_height'] = properties[0]
            layer['window_width'] = properties[1]
            layer['stride_height'] = properties[2]
            layer['stride_width'] = properties[3]
        elif section.startswith(prefixes[4]):
            # If layer is average pool
            layer['name'] = 'avg_pool'
            properties = dict[namescope]
            layer['window_height'] = properties[0]
            layer['window_width'] = properties[1]
            layer['stride_height'] = properties[2]
            layer['stride_width'] = properties[3]
        elif section.startswith(prefixes[5]):
            # If layer is a ReLU activation
            layer['name'] = 'relu'

        json_object['layers'].append(layer)

    with open("mnist_final", 'w') as outfile:
        json.dump(json_object, outfile)
        outfile.close()
