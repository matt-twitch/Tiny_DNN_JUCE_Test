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

tiny_dnn::tensor_t generatePad() // generates one sequence of type pad
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
    
    tiny_dnn::vec_t seq {attack, decay, sustain, release};
    tiny_dnn::tensor_t pad {seq};
    
    return pad;
}

tiny_dnn::tensor_t generateLead() // generates one sequence of type lead
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
    
    tiny_dnn::vec_t seq {attack, decay, sustain, release};
    tiny_dnn::tensor_t lead {seq};
    
    return lead;
}

void construct_cutoff_nn()
{
    const int num_features = 2;
    const int hidden_size = 128;
    
    // create arbitrary data
    
    
    tiny_dnn::network<tiny_dnn::sequential> net;
    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
    
    net << tiny_dnn::layers::fc(1, num_features, false, backend_type);
    net << tiny_dnn::layers::fc(num_features, hidden_size, false, backend_type);
    net << tiny_dnn::activation::relu();
    net << tiny_dnn::layers::fc(hidden_size, num_features, false, backend_type);
    net << tiny_dnn::activation::rectified_linear();
    
}

void construct_adsr_rnn()
{
    const int num_features = 4; // Number of input features
    const int hidden_size = 128; // size of hidden layers
    const int sample_size = 200; // * 2 = number of time steps
    const int test_size = 100;
 
    // training data
    std::vector<tiny_dnn::tensor_t> adsr_params;
    for(int i = 0 ; i < sample_size ; i++)
        adsr_params.push_back(generatePad());
    
    for(int i = 0 ; i < sample_size ; i++)
        adsr_params.push_back(generateLead());
    
    // (0, 1) = pad, (1, 0) = lead
    std::vector<tiny_dnn::tensor_t> insts;
    for(int i = 0 ; i < sample_size ; i++) // add pads
    {
        tiny_dnn::vec_t insts_vec {0, 1, 0, 0};
        tiny_dnn::tensor_t insts_tens {insts_vec};
        insts.push_back(insts_tens);
    }
    
    for(int i = 0 ; i < sample_size ; i++) // add labels
    {
        tiny_dnn::vec_t insts_vec {1, 0, 0, 0};
        tiny_dnn::tensor_t insts_tens {insts_vec};
        insts.push_back(insts_tens);
    }
    
    tiny_dnn::network<tiny_dnn::sequential> nn;
    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
    
    tiny_dnn::recurrent_layer_parameters params;
    params.clip = 0;
    
    int input_size = 4;
    
    nn << tiny_dnn::layers::fc(input_size, num_features, false, backend_type);
    nn << tiny_dnn::recurrent_layer(tiny_dnn::lstm(num_features, hidden_size), num_features, params);
    nn << tiny_dnn::activation::relu();
    nn << tiny_dnn::layers::fc(hidden_size, num_features, false, backend_type);
    nn << tiny_dnn::activation::softmax();
    
    tiny_dnn::adam opt;
    size_t batch_size = 64;
    int epochs = 25;
    
    DBG("start training...");
    
    nn.fit<tiny_dnn::cross_entropy_multiclass>(opt, insts, adsr_params, batch_size, epochs);
    
    DBG("training ended");
    
    tiny_dnn::vec_t input {0, 1, 0, 0};
    tiny_dnn::vec_t result = nn.predict(input);
    
    for(int i = 0 ; i < result.size() ; i++)
        DBG("result = " << result[i]);
}

void construct_cnn()
{
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

    construct_adsr_rnn();

    return 0;
}
