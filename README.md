# Gensleep
Code directory for the master thesis working with Large Scale Sleep Staging.

This code is based of the implementation created by [Supratak et al.](https://github.com/akaraspt/deepsleepnet).
## Files ##
- **trainer**, contains code for training both parts of the networks.
- **model**, contains code for the two parts of network, aswell as code for ensemble
- **utils**, code for zoneout and other stuff.
- **data_loaders**, code for data loading and preparing datasets
- **gen\_train**, Used to train a model
- **gen\_predict**, Used to predict on the same data as trained on
- **get\_perf**, calculate measures from an output folder
- **single\_predict**, predict but for one subject at a time
- **plot\_hypno**, creates hypnogram from a directory if given a subjects labels

## Setup ##
- Linux system
- Python (2.7)
- Tensorflow-GPU (1.4.0)
- Scipy (1.0.1)
- Numpy (1.14.3)
- Scikit-learn (0.19.1)
- Matplotlib (2.1.0)
