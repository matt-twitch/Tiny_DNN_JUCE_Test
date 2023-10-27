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

void construct_cnn() {

/*
    // ARCHITECTURE //
 
    // CNN for feature recognition
    Conv1 - Filter size 16
    MaxPool1 - Pool size 2
    Conv2 - Filter size 32
    MaxPool2 - Pool Size 2
    Flatten
    
*/
    
    tiny_dnn::network<tiny_dnn::sequential> net;
    
    const size_t numSamples = 100;
    const size_t numFeatures = 4;
    
    // add layers
    net << tiny_dnn::layers::conv(numSamples, numFeatures, 2, 1, 4) << tiny_dnn::activation::relu();
    net << tiny_dnn::layers::max_pool(net.in_data_size(), 1, 0, 2);
    net << tiny_dnn::layers::conv(4, 1, 2, 1, 4)  << tiny_dnn::activation::relu();
    net << tiny_dnn::layers::max_pool(net.in_data_size(), 1, 0, 2);
    net << tiny_dnn::layers::fc(4, 1, 0) << tiny_dnn::activation::relu();
    
    tiny_dnn::adam opt;
    
    // load training data
    std::vector<tiny_dnn::label_t> t_labels;
    std::vector<tiny_dnn::vec_t> t_data;
    
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
