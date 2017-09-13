// TODO: Implement memory deallocation

#include "NvInfer.h"
#include "NvUtils.h"
#include "/usr/people/bodun/include/cuda_runtime_api.h"
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
#include <json/json.h>
#include <malloc.h>
#include "include/rapidjson/document.h"
#include "include/rapidjson/stringbuffer.h"

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
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int CHANNEL_NUM = 3;
static const int OUTPUT_SIZE = 1000;
static const int VGG_MEAN[3] = {124, 117, 104};
static string DIR_PATH;

-/**
- * Implementation of the ILogger interface that prints any errors encountered during
- * network construction and inferencing.
- */
class Logger : public ILogger {
    
    void log(Severity severity, const char* msg) override {
        if (severity != Severity::kINFO) {    
            std::cout << msg << std::endl;
        }
    
    }

} gLogger;

-/**
- * Implementation of the IProfiler interface that prints the times it takes for each
- * layer to transform a tensor during inferencing.
- */
class Profiler : public IProfiler {

    void reportLayerTime(const char* layerName, float ms) override {
        ofstream o_stream("imagenet_data/imagenet_final_layer.txt", ofstream::app);
        string name = string(layerName);
        if (name == "SM_") o_stream << name << ": " << ms * 1000 << endl << endl;
        else o_stream << name << ": " << ms * 1000 << endl;
        o_stream.close();
    }

} gProfiler;

-/**
- * Stores the values of a JSON array of bias values into a vector.
- */
void parseBiases(vector<float>& vBiases, Json::Value& biases) {
    for (Json::ArrayIndex i = 0; i < biases.size(); i++) {
        vBiases.push_back(biases[i].asFloat());
    }
}

-/**
- * Stores the values of a nested JSON array of weight values into a vector.
- * 2D flattening is implemented.
- */
void parse2DWeights(vector<float>& vWeights, Json::Value& weights, int step) {
    for (Json::ArrayIndex j = 0; j < weights[0].size(); j++) {
        for (int i = 0; i < step; i++) {
            for (Json::ArrayIndex k = i; k < weights.size(); k += step) {
                vWeights.push_back(weights[k][j].asFloat());
            }
        }
    }
}

-/**
- * Stores the values of a quadrupally nested JSON array of weight values into a vector.
- * 4D flattening is implemented. Note that these weights are flattened into a KCRS format.
- */
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

-/**
- * Modifies a name until it is unique, starts tracking it, and returns it.
- */
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

-/**
- * Parses a JSON structure storing the representation of a neural network into
- * a serialized TensorRT model
- */
void APIToModel(IHostMemory **modelStream) {
    
    // create the builder
    cout << "before create builder" << endl;
    IBuilder* builder = createInferBuilder(gLogger);
    cout << "after create builder" << endl;

    // create the model to populate the network, then set the outputs and create an engine
    INetworkDefinition* network = builder->createNetwork();
   
    ifstream input(DIR_PATH + "input");
    cout << "before assert" << endl;
    assert(input.is_open() && "Unable to load Json file.");
    cout << "after assert" << endl;
    Json::CharReaderBuilder rbuilder1;
    string errs1;
    Json::Value root;
    cout << "before json parse" << endl;
    bool ok = Json::parseFromStream(rbuilder1, input, &root, &errs1);
    input.close();
    assert(ok && "Json file was unable to be parsed into a json object");

    // create inputs to model
    DimsCHW inputDims{root["num_input_channels"].asInt(), root["input_height"].asInt(), root["input_width"].asInt()};
    auto data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, inputDims);

    set<string> layer_names;
    map<string, Weights> weight_map;
    int nchw_step = 1, counter = 1;
    while (1) {
        
        string layer_path = DIR_PATH + to_string(counter);
        ifstream layer_stream(layer_path);
        if (!layer_stream.is_open()) break;
        Json::CharReaderBuilder rbuilder2;
        string errs2;
        Json::Value layer;
        bool ok = Json::parseFromStream(rbuilder2, layer_stream, &layer, &errs2);
        layer_stream.close();
        assert(ok && "Json file was unable to be parsed into a json object");

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
        counter++;

    }
    data->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*data);
    
    cout << mallinfo().hblkhd << " " << mallinfo().arena << mallinfo().fordblks << endl;
    // Build the engine
    builder->setMaxBatchSize(1);
    size_t size = 1;
    builder->setMaxWorkspaceSize(size << 25);
    auto engine = builder->buildCudaEngine(*network);
    assert(engine != nullptr);
    network->destroy();
    (*modelStream) = engine->serialize();

    // Write serialized TensorRT network to file
    //ofstream planStream("plan", ios::out | ios::binary);
    //planStream.write((char*)(*modelStream)->data(), (*modelStream)->size());
    //planStream.close();
    engine->destroy();
    builder->destroy();
}

/**
 * Reads a file of ground-truth labels into a vector of ints
 */
void readLabels(const string fileName, vector<int>& data) {
    ifstream infile(fileName);
    assert(infile.is_open() && "Unable to load label file.");
    string label;
    while (infile >> label) {
        data.push_back(stoi(label));
    }
}

/**
 * Reads in an ImageNet image with pixels stored in text form.
 * The array is transposed in order to conform with the TensorRT
 * layer weight arrangement.
 */
bool readImage(const string fileName, float* data) {
    ifstream infile(fileName);
    cout << fileName << endl;
    if (!infile.is_open()) return false;
    string word;
    vector<string> d1(0);
    vector<string> d2(0);
    vector<string> d3(0);
    while (infile >> word) {
        d1.push_back(word);
        infile >> word;
        d2.push_back(word);
        infile >> word;
        d3.push_back(word);
    }
    for (unsigned int i = 0; i < d1.size(); i++) {
        data[i] = atof(d1[i].c_str()) - VGG_MEAN[0];
    }
    for (unsigned int i = 0; i < d2.size(); i++) {
        data[i + d2.size()] = atof(d2[i].c_str()) - VGG_MEAN[1];
    }
    for (unsigned int i = 0; i < d3.size(); i++) {
        data[i + d3.size() * 2] = atof(d3[i].c_str()) - VGG_MEAN[2];
    }
    infile.close();
    return true;
}

-/**
- * Performs synchronous inference on files in a directory and stores results in a vector of outputs.
- */
void doInference(IExecutionContext& context, string dir, vector<float*>& output, int batchSize, ofstream& o_stream) {
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly 2
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    struct timeval start;
    struct timeval copy_to;
    struct timeval copy_back;
    struct timeval end;

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * CHANNEL_NUM * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    for (unsigned int i = 0; i < 50000; i++) {
        float* data = (float*)malloc(CHANNEL_NUM*INPUT_H*INPUT_W*sizeof(float));
        if(!readImage(dir + "/" + to_string(i+1), data)) {
            output.push_back((float*)malloc(sizeof(float)));
            output[i][0] = -1;
            continue;
        } 
        output.push_back((float*)malloc(OUTPUT_SIZE*sizeof(float)));

        //Start timer
        gettimeofday(&start, NULL);

        // DMA the input to the GPU, execute the batch synchronously, and DMA it back
        CHECK(cudaMemcpy(buffers[inputIndex], data, batchSize * CHANNEL_NUM * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice));
        gettimeofday(&copy_to, NULL);
        context.execute(batchSize, buffers);
        gettimeofday(&copy_back, NULL);
        CHECK(cudaMemcpy(output[i], buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        //End timer
        gettimeofday(&end, NULL);

        o_stream << "Layer " << i << ": copy_to[" << copy_to.tv_usec - start.tv_usec << "], execute[" << copy_back.tv_usec - copy_to.tv_usec << "], copy_back[" << end.tv_usec - copy_back.tv_usec << "]" << endl;
        free(data);
        cout << "inference done" << endl;
    }

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    
}

/**
 * Custom comparison struct
 */
struct Comp{
    Comp( const float* p ) : _p(p) {}
    bool operator ()(int a, int b) { return _p[a] > _p[b]; }
    const float* _p;
};

int main(int argc, char *argv[]) {
    
    //The first argument must be the name of a directory containing JSON model files.
    if (argv[1] == NULL) {
        cerr << "The first argument must be the name of a directory containing JSON model files." << endl;
        return 1;
    }
   
    //The second argument must be a directory containing ImageNet images
    if (argv[2] == NULL) {
        cerr << "The second argument must be a directory containing ImageNet images." << endl;
        return 1;
    }

    //The third argument must be the name of a file of Imagenet labels
    if (argv[3] == NULL) {
        cerr << "The third argument must be the name of a file of Imagenet labels." << endl;
        return 1;
    }

    vector<int> labelData(0);
    readLabels(string(argv[3]), labelData);

    ifstream in(string(argv[1]), ifstream::binary);
    auto const start_pos = in.tellg();
    in.ignore(numeric_limits<streamsize>::max());
    auto const char_count = in.gcount();
    in.seekg(start_pos);
    auto m = malloc(char_count);
    in.read((char*)m, char_count);

    try{

        DIR_PATH = string(argv[1]) + "/";
        IRuntime* runtime = createInferRuntime(gLogger);
        ICudaEngine* engine = runtime->deserializeCudaEngine(m, char_count, nullptr);
        IExecutionContext *context = engine->createExecutionContext();

        ofstream o_stream("imagenet_data/tensorrt_layer.txt", ofstream::trunc);
        context->setProfiler(&gProfiler);

        vector<float*> output;
        doInference(*context, string(argv[2]), output, 1, o_stream);
        o_stream.close();
        
        // destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
        
        // Determining top1 and top5
        int total = 0, top1 = 0, top5 = 0;
        for (unsigned int i = 0; i < output.size(); i++) {
            if (output[i][0] == -1) continue;
            vector<int> vx(OUTPUT_SIZE);
            for (int j = 0; j < OUTPUT_SIZE; j++) vx[j] = j;
            partial_sort(vx.begin(), vx.begin() + 5, vx.end(), Comp(output[i]));

            total++;
            if (++vx[0] == labelData[i]) {
                top1++;
                top5++;
            }
            else {
                for (int j = 1; j < 6; j++) {
                    if (++vx[j] == labelData[i]) {
                        top5++;
                        break;
                    }
                }
            }
            
        }

        ofstream accuracy("imagenet_data/imagenet_tensorrt_accuracy.txt");
        cout << "Top 1 Accuracy: " << ((float)top1) / total << endl;
        cout << "Top 5 Accuracy: " << ((float)top5) / total << endl;
        accuracy << "Total: " << total << endl;
        accuracy << "Top1Num: " << top1 << endl << "Top5Num: " << top5 << endl;
        accuracy << "Top1Prob: " << ((float)top1) / total << endl << "Top5Prob: " << ((float)top5) / total << endl;
        accuracy.close();

    } catch (cudaError e) {
        cerr << e << endl;
    }
    return 0;

}
