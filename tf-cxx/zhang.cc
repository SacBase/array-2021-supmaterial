#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/numeric_op.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/state_ops.h>
#include <tensorflow/cc/ops/random_ops.h>
#include <tensorflow/cc/ops/training_ops.h>
#include <tensorflow/cc/ops/training_ops.h>

#include <iostream>
#include <chrono>

#include "mnist/mnist_reader.hpp"

using namespace std;
using namespace tensorflow;

int main(int argc, char * argv[])
{
    SessionOptions opts;
    Scope root = Scope::NewRootScope();

    int gpu_toggle = 0; // default no gpu used
    int num_threads = 1; // default is sequential execution

    // handle arguments
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-t") {
                if (i+1 < argc) {
                    num_threads = std::stoi (argv[i+1], nullptr, 10);
                    i++; // skip this parameter
                }
                else {
                    std::cerr << "Missing value to `-t`, give a number." << std::endl;
                    return 1;
                }
            }
            else if (arg == "-g") {
                gpu_toggle = 1;
            }
            else {
                std::cerr << "Unknown parameter `" << arg << "`, we only support `-t` (num threads) and `-g` (GPU toggle)." << std::endl;
                return 1;
            }
        }
    }

    if (gpu_toggle) {
        num_threads = 1;
        LOG(INFO) << "running on gpu, so we reset num threads to 1";
    }

    LOG(INFO) << "settings are: threads(" << num_threads << "), gpu(" << gpu_toggle << ")";

    // The default template parameters are seemingly doing the right
    // job, so there is no need to define them.
    LOG(INFO) << "reading MNIST data from " << MNIST_DATA_LOCATION;
    LOG(INFO) << "MNIST data is flat, e.g images x all values";
    auto data = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t> (MNIST_DATA_LOCATION);
    // We get back a struct that contains images and lables.
    // Images are of type Image that is a 2-d array (nested vector)
    // of uint8_t respectively for training and testing images.
    // We will need to convert these to float array.
    auto trainset = data.training_images;
    auto trainlabels = data.training_labels;
    auto testset = data.test_images;
    auto testlabels = data.test_labels;

    // The number of images.
    auto sz_training_images = trainset.size ();
    LOG(INFO) << "we have " << sz_training_images << " training images";
    auto sz_training_labels = trainlabels.size ();
    LOG(INFO) << "we have " << sz_training_labels << " training labels";

    auto sz_test_images = testset.size ();
    LOG(INFO) << "we have " << sz_test_images << " test images";
    auto sz_test_labels = testlabels.size ();
    LOG(INFO) << "we have " << sz_test_labels << " test labels";

    auto x = ops::Placeholder (root, DT_FLOAT);
    auto lab = ops::Placeholder (root, DT_FLOAT);

    auto w1 = ops::Variable (root.WithOpName("w1"), {5,5,1,6}, DT_FLOAT);
    auto assign_w1 = ops::Assign (root.WithOpName("assign_w1"), w1, Input (Input::Initializer (1.0f/25.0f, TensorShape ({5,5,1,6}))));
    auto w2 = ops::Variable (root.WithOpName("w2"), {5,5,6,12}, DT_FLOAT);
    auto assign_w2 = ops::Assign (root.WithOpName("assign_w2"), w2, Input (Input::Initializer (1.0f/150.0f, TensorShape ({5,5,6,12}))));
    auto w3 = ops::Variable (root.WithOpName("w3"), {192,1,1,10}, DT_FLOAT);
    auto assign_w3 = ops::Assign (root.WithOpName("assign_w3"), w3, Input (Input::Initializer (1.0f/192.0f, TensorShape ({192,1,1,10}))));

    auto b1 = ops::Variable (root.WithOpName("b1"), {1,1,1,6}, DT_FLOAT);
    auto assign_b1 = ops::Assign (root.WithOpName("assign_b1"), b1, Input (Input::Initializer (1.0f/6.0f, TensorShape ({1,1,1,6}))));
    auto b2 = ops::Variable (root.WithOpName("b2"), {1,1,1,12}, DT_FLOAT);
    auto assign_b2 = ops::Assign (root.WithOpName("assign_b2"), b2, Input (Input::Initializer (1.0f/12.0f, TensorShape ({1,1,1,12}))));
    auto b3 = ops::Variable (root.WithOpName("b3"), {1,1,1,10}, DT_FLOAT);
    auto assign_b3 = ops::Assign (root.WithOpName("assign_b3"), b3, Input (Input::Initializer (1.0f/10.0f, TensorShape ({1,1,1,10}))));

    // begin layers
    auto tr_input = ops::Reshape (root, x, {-1, 28, 28, 1}); // reshape into 4D tensor from 2D
    // We are going to name the variables of the layers similarly to
    // SacBase/CNN, so that it is easier to cmpare the implementations.
    auto c1_ = ops::Conv2D (root, tr_input,
                              // From the docs, the filter is in the format:
                              // [filter_height, filter_width, in_channels, out_channels]
                              //Tensor(DT_FLOAT, TensorShape({5,5,1,6})),
                              w1,
                              // Strides are all units
                              {1,1,1,1},
                              // This means: ignore the boundaries
                              "VALID");
    auto a1_ = ops::Add (root.WithOpName("add_b1"), c1_, b1);
    auto c1 = ops::Sigmoid (root, a1_);
    auto s1 = ops::AvgPool (root, c1,
                            // From the docs:
                            // ksize: The size of the sliding window for each dimension of value
                            // where value has the following dimensions:
                            // [batch, height, width, channels]
                            {1, 2, 2, 1},
                            // strides:
                            {1, 2, 2, 1},
                            "VALID");

    // At this point s1 is of shape [-1,12,12,6].
    auto c2_ = ops::Conv2D (root, s1,
                            //Tensor(DT_FLOAT, TensorShape({5,5,6,12})),
                            w2,
                            {1,1,1,1},
                            "VALID");
    auto a2_ = ops::Add (root.WithOpName("add_b2"), c2_, b2);
    auto c2 = ops::Sigmoid (root, a2_);
    auto s2 = ops::AvgPool (root, c2,
                            {1, 2, 2, 1},
                            {1, 2, 2, 1},
                            "VALID");

    auto flat = ops::Reshape (root, s2,
                              // This explicit 192 is just the sanity check
                              // If the sizes wouldn't match we'll get a runtime error.
                              {-1, 192, 1, 1});

    // There is no Dense operator, so we can use either MatMul or Conv2D
    // as it is done in the cnn.sac.
    auto out_ = ops::Conv2D (root, flat,
                            //Tensor(DT_FLOAT, TensorShape({192,1,1,10})),
                            w3,
                            {1,1,1,1},
                            "VALID");
    auto a3_ = ops::Add (root.WithOpName("add_b3"), out_, b3);
    auto out = ops::Sigmoid (root, a3_);

    auto lab_reshaped = ops::Reshape (root, lab, {-1, 1, 1, 10});
    auto diff = ops::Subtract (root, out, lab_reshaped);
    auto loss = ops::L2Loss (root, diff);

    // settings
    const size_t EPOCHS = 10;
    const size_t BATCH_SIZE = 100;
    const size_t TRAINING = 10000;
    const size_t TESTING = 10000;
    const float RATE = 0.05;
    LOG(INFO) << "Settings: epochs = " << EPOCHS << ", batches = " << BATCH_SIZE
        << ", rate = " << RATE  << ", training = " << TRAINING
        << ", testing = " << TESTING;

    // some debugging
    if (!root.ok()) {
        LOG(FATAL) << root.status().ToString();
        abort();
    }

    // Now we define which weight do we want to minimise
    std::vector<Output> grad_outputs;
    TF_CHECK_OK (AddSymbolicGradients(root, {loss}, {w1, w2, w3, b1, b2, b3}, &grad_outputs));
    auto apply_w1 = ops::ApplyGradientDescent(root, w1, ops::Cast (root, RATE, DT_FLOAT), {grad_outputs[0]});
    auto apply_w2 = ops::ApplyGradientDescent(root, w2, ops::Cast (root, RATE, DT_FLOAT), {grad_outputs[1]});
    auto apply_w3 = ops::ApplyGradientDescent(root, w3, ops::Cast (root, RATE, DT_FLOAT), {grad_outputs[2]});
    auto apply_b1 = ops::ApplyGradientDescent(root, b1, ops::Cast (root, RATE, DT_FLOAT), {grad_outputs[3]});
    auto apply_b2 = ops::ApplyGradientDescent(root, b2, ops::Cast (root, RATE, DT_FLOAT), {grad_outputs[4]});
    auto apply_b3 = ops::ApplyGradientDescent(root, b3, ops::Cast (root, RATE, DT_FLOAT), {grad_outputs[5]});

    // set to use 1 core of 1 CPU
    opts.config.set_intra_op_parallelism_threads (num_threads);
    opts.config.set_inter_op_parallelism_threads (num_threads);

    // set GPU off
    auto* device_count = opts.config.mutable_device_count();
    device_count->insert({"GPU", gpu_toggle});

    // create the session with root and session options
    ClientSession session(root, opts);

    std::vector<Tensor> output;
    // init the weights and biases by running the assigns nodes once
    TF_CHECK_OK (session.Run ({assign_w1, assign_w2, assign_w3, assign_b1, assign_b2, assign_b3}, nullptr));

    auto tr_input_flat = Tensor (DT_FLOAT, TensorShape({BATCH_SIZE, 28 * 28}));
    auto input_map = tr_input_flat.tensor<float, 2>();
    auto tr_labels = Tensor (DT_FLOAT, TensorShape ({BATCH_SIZE, 10}));
    auto labels_map = tr_labels.tensor<float, 2>();

    // training
    LOG(INFO) << "Doing training";
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
        for (size_t step = 0; step < TRAINING/BATCH_SIZE; ++step) {
            size_t offset = BATCH_SIZE * step;
            for (size_t i = 0; i < BATCH_SIZE; i++) {
                for (size_t j = 0; j < trainset[offset+i].size(); j++)
                    input_map (i, j) = trainset[offset+i][j];
                for (size_t k = 0; k < 10; k++)
                    labels_map (i, k) = trainlabels[offset+i] == k ? 1.0 : 0.0;
            }

            TF_CHECK_OK (session.Run ({{x, tr_input_flat}, {lab, tr_labels}},
                                      {loss, apply_w1, apply_w2, apply_w3, apply_b1, apply_b2, apply_b3},
                                      &output));
        }
        LOG(INFO) << "Loss after " << epoch+1 << " epochs "
                  // So the loss is a L2Loss of differences between the
                  // output and the label.  Both of which are BATCH_SIZE x 10
                  // element matrices.  So if we want to have a mean error we
                  // need to divide the loss by (10 * BATCH_SIZE).
                  << output[0].scalar<float>() / 10.0f / (float)BATCH_SIZE;
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // testing
    std::vector<Tensor> eval;
    float ev_loss = 0.0;
    auto ev_input_flat = Tensor (DT_FLOAT, TensorShape({28 * 28}));
    auto ev_input_map = ev_input_flat.tensor<float, 1>();
    auto ev_labels_flat = Tensor (DT_FLOAT, TensorShape ({10}));
    auto ev_labels_map = ev_labels_flat.tensor<float, 1>();

    LOG(INFO) << "Doing testing";
    auto e1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < TESTING; i++) {
        for (size_t j = 0; j < testset[i].size(); j++) {
            ev_input_map (j) = testset[i][j];
        }
        for (size_t j = 0; j < 10; j++) {
            ev_labels_map (j) = testlabels[i] == j ? 1.0 : 0.0;
        }
        // do eval
        TF_CHECK_OK (session.Run({{x, ev_input_flat}, {lab, ev_labels_flat}}, {loss}, &eval));
        ev_loss += eval[0].scalar<float>()() / 10.0f;
    }
    auto e2 = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "Error: " << ev_loss / TESTING;
    LOG(INFO) << "Training: " << std::chrono::duration<double>(t2-t1).count() << " seconds";
    LOG(INFO) << "Testing: " << std::chrono::duration<double>(e2-e1).count() << " seconds";

    return 0;
}

