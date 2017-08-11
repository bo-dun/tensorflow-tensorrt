import tensorflow
import numpy as np
import re
import json


prefixes = ['softmax', 'fc', 'conv', 'max_pool', 'avg_pool', 'relu'] # ADD CONCAT


def parse_stride(prefix, name):
    name = name[len(prefix):]
    m1 = re.match('_\d+_\d+', name)
    if m1:
        m2 = re.findall('\d+', m1.group(0))
        return int(m2[0]), int(m2[1])

    return -1, -1


def parse_window_and_stride(prefix, name):
    name = name[len(prefix):]
    m1 = re.match('_\d+_\d+_\d+_\d+', name)
    if m1:
        m2 = re.findall('\d+', m1.group(0))
        return int(m2[0]), int(m2[1]), int(m2[2]), int(m2[3])

    return -1, -1, -1, -1


def validate_prefixes(names):
    for name in names:
        if name.startswith(prefixes[0]) or name.startswith(prefixes[1]) or name.startswith(prefixes[5]):
            continue
        elif name.startswith(prefixes[2]):
            stride = parse_stride(prefixes[2], name)
            if stride != -1:
                continue
        elif name.startswith(prefixes[3]):
            window = parse_window_and_stride(prefixes[3], name)
            if window != -1:
                continue
        elif name.startswith(prefixes[4]):
            window = parse_window_and_stride(prefixes[4], name)
            if window != -1:
                continue

        return False

    return True


def count(weight):
    if type(weight) == list:
        return sum(count(subitem) for subitem in weight)
    else:
        return 1


# Max pool must be named in the form 'max_pool_{{windowHeight}}_{{windowWidth}}_{{strideHeight}}_{{strideWidth}}' + etc.
# Average pool must be named in the form 'average_pool_{{windowHeight}}_{{windowWidth}}_{{strideHeight}}_{{strideWidth}}' + etc.
# Softmax must be named 'softmax' + etc.
# Full connected must be named 'fc' + etc.
# Convolutional must be named in the form 'conv_{{strideHeight}}_{{strideWidth}}' + etc.


arrangements = [[0,1,2,3],[0,1,3,2],[0,2,1,3],[0,2,3,1],[0,3,1,2],[0,3,2,1],[1,0,2,3],[1,0,3,2],[1,2,0,3],[1,2,3,0],
                [1,3,0,2],[1,3,2,0],[2,0,1,3],[2,0,3,1],[2,1,0,3],[2,1,3,0],[2,3,0,1],[2,3,1,0],[3,0,1,2],[3,0,2,1],[3,1,0,2],[3,1,2,0],[3,2,0,1],[3,2,1,0]]
# layer['weights_hwio'] = np.transpose(weight, (3, 2, 0, 1)).tolist()   # Rearrange order to be compatible with TensorRT
def convert(graph, namescopes, session, channels, height, width):
    if not validate_prefixes(namescopes):
        return None
    json_object = {}
    json_object['layers'] = []
    json_object['num_input_channels'] = channels
    json_object['input_height'] = height
    json_object['input_width'] = width
    for namescope in namescopes:
        print(namescope)
        layer = {}
        if namescope.startswith(prefixes[0]):
            layer['name'] = 'softmax'
        elif namescope.startswith(prefixes[1]):
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
        elif namescope.startswith(prefixes[2]):
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
            layer['stride_height'], layer['stride_width'] = parse_stride(prefixes[2], namescope)
        elif namescope.startswith(prefixes[3]):
            layer['name'] = 'max_pool'
            layer['window_height'], layer['window_width'], layer['stride_height'], layer['stride_width']\
                = parse_window_and_stride(prefixes[3], namescope)
        elif namescope.startswith(prefixes[4]):
            layer['name'] = 'avg_pool'
            layer['window_height'], layer['window_width'], layer['stride_height'], layer['stride_width']\
                = parse_window_and_stride(prefixes[4], namescope)
        elif namescope.startswith(prefixes[5]):
            layer['name'] = 'relu'

        json_object['layers'].append(layer)

        with open('/home/peter/Desktop/final_mnist.txt', 'w') as outfile:
            json.dump(json_object, outfile)
