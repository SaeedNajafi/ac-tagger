# Tagger


## Requirements
``Python 2.7`` and ``PyTorch`` (http://pytorch.org/)

## Training Instructions
```python tagger.py train <path to save model>```

### Example:
```sh
>python tagger.py train ./saved_models/
```

## Testing Instructions
```python tagger.py test <path to restore model> <input file path> <output file path>```

### Example:
```sh
>python tagger.py test ./saved_models/ ./data/dev.raw ./saved_models/dev.predicted
>python tagger.py test ./saved_models/ ./data/test.raw ./saved_models/test.predicted
```

## License
Everything is released under MIT license.
