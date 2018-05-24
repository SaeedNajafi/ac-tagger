# Tagger

## Requirements
``Python 2.7`` and ``PyTorch`` (http://pytorch.org/).

The model is implemented and tested on PyTorch version 0.3.1 (http://pytorch.org/docs/0.3.1/).

### Hardware Requirements
The model is fast on a GPU unit with CUDA + cuDNN deep learning libraries.

### Data Requirements
First you need to obtain word embeddings.

For English, we use 100-dimensions Glove embeddings (https://nlp.stanford.edu/projects/glove/).

The preprocessed version of the embeddings can be downloaded from the following link:
https://goo.gl/8D87oP

For German, we obtain and utilize the 64-dimensions German embeddings of https://arxiv.org/abs/1603.01360.

The preprocessed version of the embeddings can be downloaded from the following link:
https://goo.gl/U8dQAJ


## Running Configurations
All configurations are manually set via the ``config.py`` file.

## Training Instructions
```python tagger.py train <path to save model>```

### Example:
```sh
> mkdir ./saved_models
> python tagger.py train ./saved_models/
```

## Testing Instructions
```python tagger.py test <path to restore model> <input file path> <output file path>```

### Example:
```sh
> python tagger.py test ./saved_models/ ./data/dev.raw ./saved_models/dev.predicted
> python tagger.py test ./saved_models/ ./data/test.raw ./saved_models/test.predicted
```

## License
MIT license.
