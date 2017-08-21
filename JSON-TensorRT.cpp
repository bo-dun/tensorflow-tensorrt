// TODO: Implement memory deallocation

#include "NvInfer.h"
#include "NvUtils.h"
#include "/usr/local/cuda/include/cuda_runtime_api.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <exception>
#include <unistd.h>
#include <sys/time.h>
#include "json/json.h"

using namespace std;
using namespace nvinfer1;


#define CHECK(status) {                                                         \
    if (status != 0) {                                                          \
        cout << "Cuda failure: " << status;                                     \
        abort();                                                                \
    }                                                                           \
}                                                                               \


static const char* INPUT_BLOB_NAME = "input";
static const char* OUTPUT_BLOB_NAME = "output";
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;

/**
 * Implementation of the ILogger interface that prints any errors encountered during
 * network construction and inferencing.
 */
class Logger : public ILogger {
    
    void log(Severity severity, const char* msg) override {
        if (severity != Severity::kINFO) {    
            std::cout << msg << std::endl;
        }
    
    }

} gLogger;

/**
 * Implementation of the IProfiler interface that prints the times it takes for each
 * layer to transform a tensor during inferencing.
 */
class Profiler : public IProfiler {

    void reportLayerTime(const char* layerName, float ms) override {
        //ofstream o_stream("MNIST_layer_throughput.txt", ofstream::app);
        string name = string(layerName);
        //if (name == "SM_") o_stream << name << ": " << ms * 1000 << endl << endl;
        //else o_stream << name << ": " << ms * 1000 << endl;
        //o_stream.close();
    }

} gProfiler;

/**
 * The original MNIST data is stored in the opposite endian-ness to this machine.
 * This function reverses the endian-ness of an int primitive.
 */
int ReverseInt (int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

/**
 * Stores the values of a JSON array of bias values into a vector.
 */
void parseBiases(vector<float>& vBiases, Json::Value& biases) {
    for (Json::ArrayIndex i = 0; i < biases.size(); i++) {
        vBiases.push_back(biases[i].asFloat());
    }
}

/**
 * Stores the values of a nested JSON array of weight values into a vector.
 * 2D flattening is implemented.
 */
void parse2DWeights(vector<float>& vWeights, Json::Value& weights, int step) {
    for (Json::ArrayIndex j = 0; j < weights[0].size(); j++) {
        for (int i = 0; i < step; i++) {
            for (Json::ArrayIndex k = i; k < weights.size(); k += step) {
                vWeights.push_back(weights[k][j].asFloat());
            }
        }
    }
}

/**
 * Stores the values of a quadrupally nested JSON array of weight values into a vector.
 * 4D flattening is implemented. Note that these weights are flattened into a KCRS format.
 */
void parse4DWeights(vector<float>& vWeights, Json::Value& weights) {
    for (Json::ArrayIndex i = 0; i < weights.size(); i++) {
        for (Json::ArrayIndex j = 0; j < weights[0].size(); j++) {
            for (Json::ArrayIndex k = 0; k < weights[0][0].size(); k++) {
                for (Json::ArrayIndex l = 0; l < weights[0][0][0].size(); l++) {
                    vWeights.push_back(weights[i][j][k][l].asFloat());
                }
            }
        }
    }
}

/**
 * Modifies a name until it is unique, starts tracking it, and returns it.
 */
string uniqify(set<string>& layer_names, string name) {
    while (layer_names.find(name) != layer_names.end()) {
        name += "I";
    }
    layer_names.insert(name);
    return name;
}

/**
 * Creates a convolutional layer in the TensorRT model being constructed.
 */
ITensor* createConvolutional(INetworkDefinition* network, ITensor& input, Json::Value& layer, set<string>& layer_names, map<string, Weights>& weight_map) {
    string uniqueName = uniqify(layer_names, "CV_");
    int num_outputs = layer["num_outputs"].asInt();
    int filter_height = layer["filter_height"].asInt();
    int filter_width = layer["filter_width"].asInt();
    vector<float> vBiases(0);
    parseBiases(vBiases, layer["biases"]);
    float *biasVal = reinterpret_cast<float*>(malloc(vBiases.size() * sizeof(float)));
    for (unsigned int i = 0; i < vBiases.size(); i++) {
        biasVal[i] = vBiases[i];
    }
    weight_map[uniqueName + "bias"] = Weights{DataType::kFLOAT, biasVal, (long int) vBiases.size()};
    vector<float> vWeights(0);
    parse4DWeights(vWeights, layer["weights_hwio"]);
    float *weightVal = reinterpret_cast<float*>(malloc(vWeights.size() * sizeof(float)));
    for (unsigned int i = 0; i < vWeights.size(); i++) {
        weightVal[i] = vWeights[i];
    }
    weight_map[uniqueName + "weight"] = Weights{DataType::kFLOAT, weightVal, (long int) vWeights.size()};
    auto cv = network->addConvolution(input, num_outputs, DimsHW{filter_height, filter_width}, weight_map[uniqueName + "weight"], weight_map[uniqueName + "bias"]);
    assert(cv != nullptr);
    cv->setName(uniqueName.c_str());
    cv->setStride(DimsHW{layer["stride_height"].asInt(), layer["stride_width"].asInt()});
    int padHeight = layer["padding"].asInt();
    int padWidth = layer["padding"].asInt(); 
    if (padHeight == -1) {
        padHeight = (filter_height - 1) / 2;
        padWidth = (filter_width - 1) / 2;
    }
    cv->setPadding(DimsHW{padHeight, padWidth});
    cv->getOutput(0)->setName(uniqueName.c_str());
    return cv->getOutput(0);
}

/**
 * Creates a max pooling layer in the TensorRT model being constructed.
 */
ITensor* createMaxPool(INetworkDefinition* network, ITensor& input, Json::Value& layer, set<string>& layer_names) {
    string uniqueName = uniqify(layer_names, "MP_");
    int wHeight = layer["window_height"].asInt();
    int wWidth = layer["window_width"].asInt();
    int sHeight = layer["stride_height"].asInt();
    int sWidth = layer["stride_width"].asInt();
    auto mp = network->addPooling(input, PoolingType::kMAX, DimsHW{wHeight, wWidth});
    assert(mp != nullptr);
    mp->setName(uniqueName.c_str());
    mp->setStride(DimsHW{sHeight, sWidth});
    mp->getOutput(0)->setName(uniqueName.c_str());
    return mp->getOutput(0);
}

/**
 * Creates an average pooling layer in the TensorRT model being constructed.
 */
ITensor* createAvgPool(INetworkDefinition* network, ITensor& input, Json::Value& layer, set<string>& layer_names) {
    string uniqueName = uniqify(layer_names, "AP_");
    int wHeight = layer["window_height"].asInt();
    int wWidth = layer["window_width"].asInt();
    int sHeight = layer["stride_height"].asInt();
    int sWidth = layer["stride_width"].asInt();
    auto ap = network->addPooling(input, PoolingType::kAVERAGE, DimsHW{wHeight, wWidth});
    assert(ap != nullptr);
    ap->setName(uniqueName.c_str());
    ap->setStride(DimsHW{sHeight, sWidth});
    ap->getOutput(0)->setName(uniqueName.c_str());
    return ap->getOutput(0);
}

/**
 * Creates a fully connected layer in the TensorRT model being constructed.
 */
ITensor* createFullyConnected(INetworkDefinition* network, ITensor& input, Json::Value& layer, set<string>& layer_names, map<string, Weights>& weight_map, int step) {
    string uniqueName = uniqify(layer_names, "FC_");
    int num_outputs = layer["num_outputs"].asInt();
    vector<float> vWeights(0);
    parse2DWeights(vWeights, layer["weights"], step);
    float *weightVal = reinterpret_cast<float*>(malloc(vWeights.size() * sizeof(float)));
    for (unsigned int i = 0; i < vWeights.size(); i++) {
        weightVal[i] = vWeights[i];
    }
    weight_map[uniqueName + "weight"] = Weights{DataType::kFLOAT, weightVal, (long int) vWeights.size()};
    vector<float> vBiases(0);
    parseBiases(vBiases, layer["biases"]);
    float *biasVal = reinterpret_cast<float*>(malloc(vBiases.size() * sizeof(float)));
    for (unsigned int i = 0; i < vBiases.size(); i++) {
        biasVal[i] = vBiases[i];
    }
    weight_map[uniqueName + "bias"] = Weights{DataType::kFLOAT, biasVal, (long int) vBiases.size()};
    auto fc = network->addFullyConnected(input, num_outputs, weight_map[uniqueName + "weight"], weight_map[uniqueName + "bias"]); 
    assert(fc != nullptr);
    fc->setName(uniqueName.c_str());
    fc->getOutput(0)->setName(uniqueName.c_str());
    return fc->getOutput(0);
}

/**
 * Creates a softmax layer in the TensorRT model being constructed.
 */
ITensor* createSoftMax(INetworkDefinition* network, ITensor& input, set<string>& layer_names) {
    string uniqueName = uniqify(layer_names, "SM_");
    auto sm = network->addSoftMax(input);
    assert(sm != nullptr);
    sm->setName(uniqueName.c_str());
    sm->getOutput(0)->setName(uniqueName.c_str());
    return sm->getOutput(0);
}

/**
 * Creates a ReLU layer in the TensorRT model being constructed.
 */
ITensor* createReLu(INetworkDefinition* network, ITensor& input, set<string>& layer_names) {
    string uniqueName = uniqify(layer_names, "RL_");
    auto rl = network->addActivation(input, ActivationType::kRELU);
    assert(rl != nullptr);
    rl->setName(uniqueName.c_str());
    rl->getOutput(0)->setName(uniqueName.c_str());
    return rl->getOutput(0);
}

/**
 * Parses a JSON structure storing the representation of a neural network into
 * a serialized TensorRT model
 */
void APIToModel(Json::Value& root, IHostMemory **modelStream) {
    
    // create the builder 
    IBuilder* builder = createInferBuilder(gLogger);

    // create the model to populate the network, then set the outputs and create an engine
    INetworkDefinition* network = builder->createNetwork();

    // create inputs to model
    DimsCHW inputDims{root["num_input_channels"].asInt(), root["input_height"].asInt(), root["input_width"].asInt()};
    auto data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, inputDims);

    set<string> layer_names;
    map<string, Weights> weight_map;
    int nchw_step = 1;
    Json::Value layers = root["layers"];
    for (Json::ArrayIndex i = 0; i < layers.size(); i++) {
        Json::Value layer = layers[i];
        string name = layer["name"].asString();
        if (name == "conv") {
            data = createConvolutional(network, *data, layer, layer_names, weight_map);
            nchw_step = layer["num_outputs"].asInt();
        }
        else if (name == "max_pool") data = createMaxPool(network, *data, layer, layer_names);
        else if (name == "avg_pool") data = createAvgPool(network, *data, layer, layer_names);
        else if (name == "fc") {
            data = createFullyConnected(network, *data, layer, layer_names, weight_map, nchw_step);
            nchw_step = 1;
        }
        else if (name == "softmax") data = createSoftMax(network, *data, layer_names);
        else if (name == "relu") data = createReLu(network, *data, layer_names);
        cout << name << endl;
    }
    data->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*data);

    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 25);

    auto engine = builder->buildCudaEngine(*network);
    assert(engine != nullptr);
    network->destroy();
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}



/**
 * Opens a file storing MNIST images and returns the images stored in a vector
 * of pointers.
 */
void readMNISTImages(const std::string& fileName, vector<uint8_t*>& data) {
    ifstream infile(fileName, ifstream::binary);
    assert(infile.is_open() && "Unable to load image file."); 
    int magic, image_num, rows, cols;
    infile.read((char*)&magic, 4);
    magic = ReverseInt(magic);
    infile.read((char*)&image_num, 4);
    image_num = ReverseInt(image_num);
    infile.read((char*)&rows, 4);
    rows = ReverseInt(rows);
    infile.read((char*)&cols, 4);
    cols = ReverseInt(cols);
    cout << "Magic number: " << magic << endl;
    cout << "Number of images: " << image_num << endl;
    cout << "Number of rows: " << rows << endl;
    cout << "Number of columns: " << cols << endl;
    for (int i = 0; i < image_num; i++) {
        uint8_t* buffer = (uint8_t*)malloc(INPUT_H*INPUT_W*sizeof(uint8_t));
        infile.read(reinterpret_cast<char*>(buffer), INPUT_H*INPUT_W);
        data.push_back(buffer);
    }
    infile.close();
}

/**
 * Opens a file storing MNIST labels and returns the labels stored in a vector
 * of pointers.
 */
void readMNISTLabels(const std::string& fileName, vector<int>& labels) {
    ifstream infile(fileName, ifstream::binary);
    assert(infile.is_open() && "Unable to load label file."); 
    int magic, label_num;
    infile.read((char*)&magic, 4);
    magic = ReverseInt(magic);
    infile.read((char*)&label_num, 4);
    label_num = ReverseInt(label_num);
    cout << "Magic number: " << magic << endl;
    cout << "Number of labels: " << label_num << endl;
    for (int i = 0; i < label_num; i++) {
        uint8_t label;
        infile.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back((int)label);
    }
    infile.close();
}

/**
 * Performs synchronous inference on a vector of inputs and stores results in a vector of outputs.
 */
void doInference(IExecutionContext& context, vector<float*>& input, vector<float*>& output, int batchSize, ofstream& o_stream) {
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    
    struct timeval start;
    struct timeval end;

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
     
    for (unsigned int i = 0; i < input.size(); i++) {
        output.push_back((float*)malloc(OUTPUT_SIZE*sizeof(float)));
        //Start timer
        gettimeofday(&start, NULL);

        // DMA the input to the GPU, execute the batch asynchronously, and DMA it back
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input[i], batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.execute(batchSize, buffers);
        CHECK(cudaMemcpyAsync(output[i], buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        
        //End timer
        gettimeofday(&end, NULL);
        o_stream << "Layer " << i << ": " << end.tv_usec - start.tv_usec << endl;
    }

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    
}

int main(int argc, char *argv[]) {
    
    //The first argument must be the name of a JSON model file to read
    if (argv[1] == NULL) {
        cerr << "The first argument must be the name of a JSON file to load." << endl;
        return 1;
    }
   
    //The second argument must be the name of a file containing MNIST labels
    if (argv[2] == NULL) {
        cerr << "The second argument must be the name of a file containing MNIST labels." << endl;
        return 1;
    }

    //The third argument must be the name of a file containing MNIST images
    if (argv[3] == NULL) {
        cerr << "The third argument must be the name of a file containing MNIST images." << endl;
        return 1;
    }

    ifstream input(argv[1]);
    assert(input.is_open() && "Unable to load Json file.");
    Json::CharReaderBuilder rbuilder;
    string errs;
    Json::Value root;
    bool ok = Json::parseFromStream(rbuilder, input, &root, &errs);
    input.close();
    assert(ok && "Json file was unable to be parsed into a json object");
    
    vector<int> labelData(0);
    readMNISTLabels(string(argv[2]), labelData);
    vector<uint8_t*> imageData(0);
    readMNISTImages(string(argv[3]), imageData);
    
    vector<float*> data(0);
    for (unsigned int i = 0; i < imageData.size(); i++) {
        float* buffer = (float*)malloc(INPUT_H*INPUT_W*sizeof(float));
        for (int j = 0; j < INPUT_H*INPUT_W; j++) {
            buffer[j] = float(imageData[i][j]);
            //For a visual ascii representation of images
            //cout << (" .:-=+*#%@"[(imageData[i][j]) / 26]) << (((j+1)%INPUT_W) ? "" : "\n"); 
        }
        data.push_back(buffer);
    }
    
    IHostMemory *modelStream{nullptr};
    try{

    	APIToModel(root, &modelStream);
        IRuntime* runtime = createInferRuntime(gLogger);
        ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
        if (modelStream) modelStream->destroy();
        IExecutionContext *context = engine->createExecutionContext();

        ofstream o_stream("test_throughput.txt", ofstream::trunc);
        context->setProfiler(&gProfiler);

        vector<float*> output;
        doInference(*context, data, output, 1, o_stream);
        o_stream.close();
        
        // destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
        
        int total_correct = 0;
        int total_incorrect = 0;
        for (unsigned int i = 0; i < output.size(); i++) {
            float val = 0;
            int result = -1;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                val = max(val, output[i][j]);
                if (val == output[i][j]) result = j;
            }
            if (result == -1) {
                cerr << "There was an error performing MNIST inferencing." << endl;
            }
            if (result == labelData[i]) total_correct++;
            else total_incorrect++;
        }
        cout << "Accuracy: " << ((float)total_correct) / (total_correct + total_incorrect) << endl;

    } catch (cudaError e) {
        cerr << e << endl;
    }
    return 0;

}
