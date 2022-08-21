This is the winning solution to the DAC4AutoML Competition ([DAC4SGD track](https://github.com/automl/DAC4SGD)) hosted on Codalab. The goal was to dynamically adapt the learning rate of an SGD optimizer. The solution is pretty simple: a linear decay is applied to the learning rate at each batch step. Then, at the end of each epoch, if the validation loss worsens compared to the previous epoch, the learning rate is further reduced by a constant factor.

The directory contains the following files:
- `metadata` is an empty file required by the Codalab platform
- `requirements.txt` is empty since no additional packages were required
- `solution.py` contains the actual solution code
- the `LICENSE` file

You will also need to follow the instructions provided by the [DAC4SGD track](https://github.com/automl/DAC4SGD) page.
