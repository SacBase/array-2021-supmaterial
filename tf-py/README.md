Tensorflow CNN
--------------

We provide three implementations of the CNN:

* zhang.py --- makes use of Tensorflow 1.12.0 for Python
* zhang-tf2.py -- makes use of Tensorflow 2.2.0 for Python (relies on V1 API)
* zhang-tf2-refa.py -- makes use of Tensorflow 2.2.0 for Python (includes larger CNN network *and* use latest V2 API)

To use, one needs the `python-mnist` PyPi package installed.

The last implementation, zhang-tf2-refa, is used for our experiments and whose measurements we used in the paper.
