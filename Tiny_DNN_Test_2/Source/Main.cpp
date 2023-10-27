/*
  ==============================================================================

    This file contains the basic startup code for a JUCE application.

  ==============================================================================
*/

#include <JuceHeader.h>
#include <tiny_dnn.h>
#include "rapidcsv.h"

//==============================================================================

//std::vector<tiny_dnn::label_t> parse_Labels

void construct_rnn() {
    
    tiny_dnn::network<tiny_dnn::sequential> net;
    
    // training data (84 x 4)
    const size_t numSamples = 84;
    const size_t numFeatures = 4;
    
    // load training data
    std::vector<tiny_dnn::label_t> t_labels;
    std::vector<tiny_dnn::vec_t> t_data;
    
    tiny_dnn::network<tiny_dnn::sequential> RegNet;

    const size_t num_features = 4; // Number of input features
    const size_t num_adsr_parameters = 4; // Number of ADSR parameters
    const size_t num_classes = 2; // Number of classes (lead or pad)

    RegNet << tiny_dnn::layers::fc(num_features, 64) << tiny_dnn::activation::relu()
        << tiny_dnn::layers::fc(64, 64) << tiny_dnn::activation::relu()
        << tiny_dnn::layers::fc(64, num_adsr_parameters)
        << tiny_dnn::layers::linear(4)
        << tiny_dnn::layers::softmax_layer();
    
}

void construct_cnn() {
    
    tiny_dnn::network<tiny_dnn::sequential> net;
    
    // training data (84 x 4)
    const size_t numSamples = 84;
    const size_t numFeatures = 4;
    
    // load training data
    std::vector<tiny_dnn::label_t> t_labels;
    std::vector<tiny_dnn::vec_t> t_data;
    
    // add layers
    // width, height, kernel, input, output
    net << tiny_dnn::layers::conv(4, 84, 5, 4, 16) << tiny_dnn::activation::relu();
    net << tiny_dnn::layers::max_pool(2, 82, 16, 2);
    net << tiny_dnn::layers::conv(4, 1, 2, 1, 4)  << tiny_dnn::activation::relu();
    net << tiny_dnn::layers::ave_pool(28, 28, 6, 2);
    net << tiny_dnn::layers::fc(4, 1, 0) << tiny_dnn::activation::relu();
    
    tiny_dnn::adam opt;
    
    
/*
    // add layers
    net << conv(32, 32, 5, 1, 6) << tiny_dnn::activation::tanh()  // in:32x32x1, 5x5conv, 6fmaps
    << ave_pool(28, 28, 6, 2) << tiny_dnn::activation::tanh() // in:28x28x6, 2x2pooling
    << fc(14 * 14 * 6, 120) << tiny_dnn::activation::tanh()   // in:14x14x6, out:120
        << fc(120, 10);                     // in:120,     out:10

    assert(net.in_data_size() == 32 * 32);
    assert(net.out_data_size() == 10);

    // load MNIST dataset
    std::vector<label_t> train_labels;
    std::vector<vec_t> train_images;

    parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);

    // declare optimization algorithm
    adagrad optimizer;

    // train (50-epoch, 30-minibatch)
    net.train<mse, adagrad>(optimizer, train_images, train_labels, 30, 50);

    // save
    net.save("net");

    // load
    // network<sequential> net2;
    // net2.load("net");
 */
}


int main (int argc, char* argv[])
{

    construct_cnn();


    return 0;
}
