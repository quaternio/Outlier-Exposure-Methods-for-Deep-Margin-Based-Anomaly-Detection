'''
Of the ID classes (which form a single high-level partition), we need 3 subsets:
    ID1) ID classes
        a) Training ID
        b) Validation ID
        c) Test ID

Of the OOD classes, we have 3 high-level partitions:
    OOD1) Training OOD Classes (OE data)
    OOD2) Validation OOD Classes
    OOD3) Test OOD Classes

Question: Should training OOD classes show up in validation set or test set?

Answer:   No. I don't think they should. At each phase, we want to present the network with previously unseen anomaly types.
          Because of this, we can create 3 monolithic datasets from OOD classes.

To brainstorm a bit... 
    1) We could take CIFAR-100 and split it into its train and test parts
    2) For both parts, build all four partitions. You should have 8 partitions in total.
    3) Two of these partitions will be ID partitions: Raw Training ID and Test ID
        a) Partition Raw Training ID into an 80/20 split. Call the 80% partition the Training ID partition and
           the 20% partition the Validation ID partition. At this point, you will have 9 total partitions; three of 
           which will be for ID data.
    4) You still have 6 OOD partitions... what do we do with these? Throw out the Training OOD class "test" set. This will create a more 
       class-balanced training process. Concatenate the sets corresponding to validation OOD classes and to test OOD classes. 
       You should now have 6 partitions.
    5) Proceed!

'''

import torch
import random
import pickle as pkl
import numpy as np
import copy

import torchutil.data
import torchutil
from adtools.data.split import ClassSplit
from random_split_generator import FourWayClassSplit
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split

def build_split_datasets(split):
    assert(split < 5 and split >= 0)
    with open('id-ood-splits/split{}.pkl'.format(split), 'rb') as f:
        id_ood_split = pkl.load(f) # Type is adtools.data.FourWayClassSplit

    # Wrapper around torchvision.datasets.CIFAR100 which provides an interface for
    # getting image labels without loading images from disk (significantly speeds
    # up ID/OOD splitting process)
    train_dataset = torchutil.data.CIFAR100(root = 'data', train = True, download = True)
    test_dataset = torchutil.data.CIFAR100(root = 'data', train = False, download = True)

    id_train_dataset, ood_train_dataset, ood_val1_dataset, ood_test1_dataset = id_ood_split.split_dataset(train_dataset)
    id_test_dataset, _, ood_val2_dataset, ood_test2_dataset = id_ood_split.split_dataset(test_dataset)

    # Map ID labels and OOD labels to [0, 1, 2, ...]
    id_train_dataset = torchutil.data.TransformingDataset(
        id_train_dataset,
        target_transform = torchutil.data.LabelMappingTransform(
            label_list = id_ood_split.id_labels()
        )
    )

    id_train_len = round(0.8 * len(id_train_dataset))
    id_val_len   = len(id_train_dataset) - id_train_len

    id_train_dataset, id_val_dataset = random_split(id_train_dataset, 
                                            [id_train_len, id_val_len], 
                                            generator=torch.Generator().manual_seed(42))

    id_test_dataset = torchutil.data.TransformingDataset(
        id_test_dataset,
        target_transform = torchutil.data.LabelMappingTransform(
            label_list = id_ood_split.id_labels()
        )
    )

    ood_train_dataset = torchutil.data.TransformingDataset(
        ood_train_dataset,
        target_transform = torchutil.data.LabelMappingTransform(
            label_list = id_ood_split.ood_train_labels()
        )
    )

    ood_val1_dataset = torchutil.data.TransformingDataset(
        ood_val1_dataset,
        target_transform = torchutil.data.LabelMappingTransform(
            label_list = id_ood_split.ood_val_labels()
        )
    )

    ood_test1_dataset = torchutil.data.TransformingDataset(
        ood_test1_dataset,
        target_transform = torchutil.data.LabelMappingTransform(
            label_list = id_ood_split.ood_test_labels()
        )
    )

    ood_val2_dataset = torchutil.data.TransformingDataset(
        ood_val2_dataset,
        target_transform = torchutil.data.LabelMappingTransform(
            label_list = id_ood_split.ood_val_labels()
        )
    )

    ood_test2_dataset = torchutil.data.TransformingDataset(
        ood_test2_dataset,
        target_transform = torchutil.data.LabelMappingTransform(
            label_list = id_ood_split.ood_test_labels()
        )
    )

    ood_val_dataset  = ConcatDataset([ood_val1_dataset, ood_val2_dataset])
    ood_test_dataset = ConcatDataset([ood_test1_dataset, ood_test2_dataset])

    return (id_train_dataset, id_val_dataset, id_test_dataset), (ood_train_dataset, ood_val_dataset, ood_test_dataset)