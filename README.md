# Tensorflow to TensorRT Model Converter

Uses a JSON intermediate to port model information.
Existing JSON-TensorRT converter customized for ImageNet inferencing.

## Files
[Tensorflow-JSON.py](https://github.com/BoyDun/tensorflow-tensorrt/blob/master/Tensorflow-JSON.py) takes a TensorFlow Session and a few other related objects and serializes them to JSON format. In the code I included two variants on the serialization method, one that aggregates the entire network into one JSON object and one that serializes each layer individually. When using, make sure that the dictionary is formatted correctly according to my comments in the file.

[JSON-TensorRT.cpp](https://github.com/BoyDun/tensorflow-tensorrt/blob/master/JSON-TensorRT.cpp)
