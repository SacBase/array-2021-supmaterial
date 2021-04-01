ARRAY 2021 Supplementary Materials
==================================

About
-----

This directory contain multiple implementations of a CNN for handwritten
image recognition that is based on Zhifei Zhang's paper *Derivation of
Backpropagation in Convolutional Neural Network* (see the [paper][zhang] for more details).

These implementations are presented in the paper *Array Languages Make Neural Networks Fast*,
which is submitted for review at PLDI's ARRAY 2021 Workshop.

We have four implementations, one in Tensorflow (Python), TensorFlow (C++), PyTorch (Python),
SaC. These implementations use the MNIST dataset (see the `input` directory for details on
how to set this up).

The layout of this repo is as follows:

```
 input
 └── README.md               # A directory for MNIST inputs
 pt
 ├── main.py                 # PyTorch version of Zhang's CNN
 └── README.md
 sac
 ├── cnn.sac                 # Building blocks written using with-loops
 ├── cnn_tc.sac              # Building blocks using new notation as in the paper
 ├── mnist.sac               # Helper module to read in the data
 └── zhang.sac               # The actual CNN
 tf-cxx
 ├── cmake                   # Helper CMake modules
 ├── CMakeLists.txt          # CMake file to gather all the dependencies for
 │                           # the dependent libraries
 ├── README.md
 └── zhang.cc                # The C++ version of the CNN with Tensorflow
 tf-py
 └── zhang.py                # The python version of the CNN with Tensorflow
```

Extra Details
-------------

### TensorFlow (Python)

**NOTE**: In SaC terminology, an EPOCH and STEP are **not** the same as in the
          TF terminology. In TF an epoch is related to number of data items being
          computer over, and not the number of training cycles (as it is in SaC).
          As such, for TF, `num_epoch=None`. This also means that training steps
          is different here as well: `step = SAC_STEP/BATCH_SIZE * SAC_EPOCH`, or
          with a concrete example, to get TF to perform 20 epochs, with batch size
          100, wih 10000 steps, you need to do 2000 steps in TF.


### TensorFlow (C++)

#### Compile

Create shared tensorflow libraries for C++ API:

```sh
cd <tensorflow-src>
./configure
bazel build -c opt //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //tensorflow:install_headers
```

If during `./configure` you specified CUDA support, this will
be compiled into the shared libraries. You will need to link you
TF application with CUDA/CUDNN libraries.

*NOTE*: `//tensorflow:install_headers` is only available with [TF 1.12.0][tfcxx] (and onward).

Then build the CNN:

```sh
mkdir build
cd build
cmake ..
make
```

and run `./zhang-cnn`.

PyTorch (Python)
-----

The MNIST dataset is loaded through the `torchvision.dataset.MNIST` class, which ordinarily
downloads the dataset - this is disabled. Remember to correct fill the `input` directory
*before* running the NN.

[zhang]: https://web.archive.org/web/20201207140915/http://web.eecs.utk.edu/~zzhang61/docs/reports/2016.10%20-%20Derivation%20of%20Backpropagation%20in%20Convolutional%20Neural%20Network%20(CNN).pdf
[paper]: https://arxiv.org/abs/1912.05234
[tfcxx]: https://github.com/tensorflow/tensorflow/commit/39e324505c380c9d449dc31d34629a9d470c765f
