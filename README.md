
<img src="magenta-logo-bg.png" height="75">

**Magenta** is a research project exploring the role of machine learning
in the process of creating art and music. This Fork was made to build on top of the coconet music generation model and expand it.

Notably, the ability to train a model from a folder of midi files has been developed and easy management through a GUI.

## Getting Started

* [Requirements & Installation](#installation)
* [Using the coconet GUI training](#using-coconet-gui)

## Installation

We recommend to create an anaconda environment (python 3.5-3.7).
```bash
conda create -n coconet python=3.7
```
Install the package from github per:
```bash
conda activate coconet
pip install https://github.com/sleep-yearning/magenta/archive/master.zip
```
 
If you want to enable GPU support and [can run it](
https://www.tensorflow.org/install/install_linux#nvidia_requirements_to_run_tensorflow_with_gpu_support), you should install 
the magenta-gpu package //TODO: insert branch link command

The only difference between the two packages is that `magenta-gpu` depends on
`tensorflow-gpu` instead of `tensorflow`.

**NOTE**: In order to install the `rtmidi` package that magenta depends on, you may need to install headers for some sound libraries. On Ubuntu, these are the necessary packages:
- build-essential
- libasound2-dev
- libjack-dev

On Arch based systems, those packages are:
- base-devel
- jack
- alsa-lib

If you run into problems with this package, you can try and install a copy of our package without the requirement (found in setup.py). This might work since coconet doesn't really use it, but will probably break if you want to use the same magenta installation for something else. You specifically won't be able to use magentas midi-interface for playback.


## Using CocoNet GUI

You can start the GUI with: 

```bash
python 
from magenta.models.coconet import GUI
```

From there you can either select your folder of midi files to train on, or generate new midi files from already trained models.

Alternatively just call the python files in the install directory with their needed/optional arguments.
For example:
```bash
python /magenta/models/coconet/prepare_and_train.py path-to-midi-folder --grouped
python /magenta/models/coconet/coconet_sample.py path-to-trained-model-folder igibbs midi-output-folder
```

If you want to use the package more flexible and specify more hyperparameters without the GUI, you can use
```bash
from magenta.models.coconet.prepare_and_train import prepare, train
from magenta.models.coconet.coconet_sample import main
```
and then call those functions in your code.


The original magenta implementation can be found [here](https://github.com/tensorflow/magenta)
This version was forked from [here](https://github.com/everettk/magenta) because necessary changes for CocoNet to work with python 3 where added there.
