# Tensorflow to TensorRT Model Converter

This conversion implementation uses a JSON intermediate to port neural network information. Any TensorFlow model can be converted to TensorRT format, optimized, and used for highly efficient inferencing. The current converter from JSON to TensorRT includes customized code for loading ImageNet data for inferencing.

## Files
[Tensorflow-JSON.py](https://github.com/BoyDun/tensorflow-tensorrt/blob/master/Tensorflow-JSON.py) takes a TensorFlow Session and a few other related objects and serializes them to JSON format. In the code I included two variants on the serialization method, one that aggregates the entire network into one JSON object and one that serializes each layer individually. When using, make sure that the dictionary is formatted correctly according to my comments in the file.

[JSON-TensorRT.cpp](https://github.com/BoyDun/tensorflow-tensorrt/blob/master/JSON-TensorRT.cpp) deserializes a directory of JSON layer files into a TensorRT model. This file also includes customized code to load and synchronously evaluate ImageNet data for direct inferencing. If the input data you're loading has multiple channels and/or batches, it must be in NCHW format, which is what TensorRT works with. TensorFlow, however, uses NHWC format. When debugging, keep in mind that convolutional weights are in KCRS format in TensorRT and RSCK format in TensorFlow.

## Author
* Peter Dun, bodun@stanford.edu

Feel free to reach out with any questions, comments, suggestions, bug reports, etc.
## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/BoyDun/tensorflow-tensorrt/blob/master/LICENSE) file for more details.
