# Bugs and Bug Fixes
- **Problem**: Unable to quantize buffer or min/max value for input 1 in op MUL in subgraph 0, node: 8 in TF 1.14 compiled from source
    - **Solution**: `sudo -H pip install tensorflow-gpu=1.15` [Source](https://github.com/tensorflow/tensorflow/issues/30838)
- **Problem**: TF Lite conversion of minimal graph with tf.matmul fails on Linux
    - **Solution**: Compile 1.14 From Source on ubuntu 18.04 or `sudo -H pip install tensorflow-gpu=1.15` [Source](https://github.com/tensorflow/tensorflow/issues/27640)
- **Problem**: No module named `tflite_runtime`
    - **Solution**: Install from [here](https://www.tensorflow.org/lite/guide/python). Only works in Python3.