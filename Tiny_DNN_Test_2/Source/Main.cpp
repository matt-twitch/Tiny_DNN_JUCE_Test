/*
  ==============================================================================

    This file contains the basic startup code for a JUCE application.

  ==============================================================================
*/

#include <JuceHeader.h>
#include <tiny_dnn.h>
#include "csv.h"

//==============================================================================


float roundFloat(float toRound)
{
    float val = (int)(toRound * 100 + .5);
    return (float)val / 100;
}

tiny_dnn::vec_t generatePad()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_real_distribution<> att(0.3, 0.9);
    std::uniform_real_distribution<> dec(0.3, 0.8);
    std::uniform_real_distribution<> sus(0.4, 0.7);
    std::uniform_real_distribution<> rel(0.6, 0.9);
    
    float attack = roundFloat(att(gen));
    float decay = roundFloat(dec(gen));
    float sustain = roundFloat(sus(gen));
    float release = roundFloat(rel(gen));
    
    tiny_dnn::vec_t pad {attack, decay, sustain, release};
    
    return pad;
}

tiny_dnn::vec_t generateLead()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_real_distribution<> att(0.1, 0.3);
    std::uniform_real_distribution<> dec(0.2, 0.5);
    std::uniform_real_distribution<> sus(0.1, 0.3);
    std::uniform_real_distribution<> rel(0.1, 0.3);
    
    float attack = roundFloat(att(gen));
    float decay = roundFloat(dec(gen));
    float sustain = roundFloat(sus(gen));
    float release = roundFloat(rel(gen));
    
    tiny_dnn::vec_t lead {attack, decay, sustain, release};
    
    return lead;
}

void construct_rnn()
{
    const int num_features = 4; // Number of input features, equivalent to sequence length
    const int num_vals = 4; // Number of possible values, equivalent to vocab size
    const int hidden_size = 128; // size of hidden layers
    const int sample_size = 200;
    
    std::vector<tiny_dnn::vec_t> values;
    for(int i = 0 ; i < sample_size ; i++)
        values.push_back(generatePad());
    
    for(int i = 0 ; i < sample_size ; i++)
        values.push_back(generateLead());
    
    // 0 = pad, 1 = lead
    std::vector<tiny_dnn::label_t> labels;
    for(int i = 0 ; i < sample_size ; i++)
        labels.push_back(0);
    
    for(int i = 0 ; i < sample_size ; i++)
        labels.push_back(1);
    
    tiny_dnn::network<tiny_dnn::sequential> nn;
    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
    
    tiny_dnn::recurrent_layer_parameters params;
    params.clip = 0;
    
    int input_size = num_vals;
    
    nn << tiny_dnn::layers::fc(num_vals, input_size, false, backend_type);
    nn << tiny_dnn::recurrent_layer(tiny_dnn::lstm(input_size, hidden_size), num_features, params);
    nn << tiny_dnn::activation::relu();
    nn << tiny_dnn::layers::fc(hidden_size, num_vals, false, backend_type);
    nn << tiny_dnn::activation::softmax();
    
    tiny_dnn::adam opt;
    size_t batch_size = 1;
    int epochs = 30;
    
    nn.train<tiny_dnn::mse>(opt, values, labels, batch_size, epochs);
    
    tiny_dnn::vec_t input {0.2, 0.4, 0.6, 0.3};
    tiny_dnn::vec_t result = nn.predict(input);
    
    for(int i = 0 ; i < result.size() ; i++)
        DBG("result " << i << " = " << result[i]);
    
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
    
}


int main (int argc, char* argv[])
{

    construct_rnn();

    return 0;
}
