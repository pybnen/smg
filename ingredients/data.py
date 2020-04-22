import numpy as np
from torch.utils.data import Subset

from sacred import Ingredient

data_ingredient = Ingredient('data')

@data_ingredient.capture
def dataset_train_valid_split(dataset, valid_split):
    ds_size = len(dataset)
    indices = np.arange(ds_size)
    train_size = int(ds_size * (1 - valid_split))

    ds_train = Subset(dataset, indices[:train_size])
    ds_valid = Subset(dataset, indices[train_size:])

    assert(len(ds_train) + len(ds_valid) == ds_size)

    return ds_train, ds_valid
