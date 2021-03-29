SaC CNN Library and Example Application
=======================================

About
-----

This work is based
on Zhifei Zhang's paper _Derivation of Backpropagation in Convolutional
Neural Network_ (see the [paper][1] for more details).

Convolutional Neural Networks
------

The file `cnn.sac` implements the various network components to construct a CNN.

In `zhang.sac` we implementat a CNN that evaluates the MNIST/EMNIST datasets. We read in MNIST/EMNIST
datasets using functions defined in `mnist.sac`.

To compile, simply call `make` which will create binaries for x64 sequential and multi-threaded.

[1]: https://web.archive.org/web/20171209050606/http://web.eecs.utk.edu/~zzhang61/docs/reports/2016.10%20-%20Derivation%20of%20Backpropagation%20in%20Convolutional%20Neural%20Network%20(CNN).pdf
