import torch
import random
import pickle as pkl
import numpy as np
import copy

import torchutil.data
import torchutil
from adtools.data.split import ClassSplit

class FourWayClassSplit:
    """
    Parameters:
        id_labels: List of in-distribution class labels.
        ood_train_labels: List of out-of-distribution oe class labels.
        ood_val_labels: List of out-of-distribution validation labels.
        ood_test_labels: List of out-of-distribution test labels.
    """
    def __init__(self, id_labels, ood_train_labels, ood_val_labels, ood_test_labels):
        self._id_labels = id_labels
        self._ood_train_labels = ood_train_labels
        self._ood_val_labels = ood_val_labels
        self._ood_test_labels = ood_test_labels

    """
    Description: Partitions a given dataset into an in-distribution dataset and
        three out-of-distribution datasets

    Parameters:
        dataset: The dataset to partition.
    """
    def split_dataset(self, dataset):
        id_indices = torchutil.data.get_indices_of_labels(
            dataset,
            self._id_labels)
        ood_train_indices = torchutil.data.get_indices_of_labels(
            dataset,
            self._ood_train_labels)
        ood_val_indices = torchutil.data.get_indices_of_labels(
            dataset,
            self._ood_val_labels)
        ood_test_indices = torchutil.data.get_indices_of_labels(
            dataset,
            self._ood_test_labels)
        
        id_dataset = torchutil.data.Subset(
            dataset,
            id_indices
        )
        ood_train_dataset = torchutil.data.Subset(
            dataset,
            ood_train_indices
        )
        ood_val_dataset = torchutil.data.Subset(
            dataset,
            ood_val_indices
        )
        ood_test_dataset = torchutil.data.Subset(
            dataset,
            ood_test_indices
        )
        
        return id_dataset, ood_train_dataset, ood_val_dataset, ood_test_dataset

    def id_labels(self):
        return copy.deepcopy(self._id_labels)
    
    def ood_train_labels(self):
        return copy.deepcopy(self._ood_train_labels)

    def ood_val_labels(self):
        return copy.deepcopy(self._ood_val_labels)

    def ood_test_labels(self):
        return copy.deepcopy(self._ood_test_labels)


num_classes = 100
num_id = 10
num_ood_train = 10
num_ood_val = 10
num_ood_test = num_classes - num_id - num_ood_train - num_ood_val

random.seed(0)

id_split_0  = random.sample(range(num_classes), num_id)
remaining = [i for i in range(num_classes) if i not in id_split_0]
ood_train_split_0 = random.sample(remaining, num_ood_train)
remaining = [i for i in remaining if i not in ood_train_split_0]
ood_val_split_0 = random.sample(remaining, num_ood_val)
ood_test_split_0 = [i for i in remaining if i not in ood_val_split_0]

id_ood_four_way_split_0 = FourWayClassSplit(id_split_0, ood_train_split_0, ood_val_split_0, ood_test_split_0)

with open("id-ood-splits/split0.pkl", "wb") as f:
   pkl.dump(id_ood_four_way_split_0, f)


id_split_1  = random.sample(range(num_classes), num_id)
remaining = [i for i in range(num_classes) if i not in id_split_1]
ood_train_split_1 = random.sample(remaining, num_ood_train)
remaining = [i for i in remaining if i not in ood_train_split_1]
ood_val_split_1 = random.sample(remaining, num_ood_val)
ood_test_split_1 = [i for i in remaining if i not in ood_val_split_1]

id_ood_four_way_split_1 = FourWayClassSplit(id_split_1, ood_train_split_1, ood_val_split_1, ood_test_split_1)

with open("id-ood-splits/split1.pkl", "wb") as f:
   pkl.dump(id_ood_four_way_split_1, f)


id_split_2  = random.sample(range(num_classes), num_id)
remaining = [i for i in range(num_classes) if i not in id_split_2]
ood_train_split_2 = random.sample(remaining, num_ood_train)
remaining = [i for i in remaining if i not in ood_train_split_2]
ood_val_split_2 = random.sample(remaining, num_ood_val)
ood_test_split_2 = [i for i in remaining if i not in ood_val_split_2]

id_ood_four_way_split_2 = FourWayClassSplit(id_split_2, ood_train_split_2, ood_val_split_2, ood_test_split_2)

with open("id-ood-splits/split2.pkl", "wb") as f:
   pkl.dump(id_ood_four_way_split_2, f)


id_split_3  = random.sample(range(num_classes), num_id)
remaining = [i for i in range(num_classes) if i not in id_split_3]
ood_train_split_3 = random.sample(remaining, num_ood_train)
remaining = [i for i in remaining if i not in ood_train_split_3]
ood_val_split_3 = random.sample(remaining, num_ood_val)
ood_test_split_3 = [i for i in remaining if i not in ood_val_split_3]

id_ood_four_way_split_3 = FourWayClassSplit(id_split_3, ood_train_split_3, ood_val_split_3, ood_test_split_3)

with open("id-ood-splits/split3.pkl", "wb") as f:
   pkl.dump(id_ood_four_way_split_3, f)


id_split_4  = random.sample(range(num_classes), num_id)
remaining = [i for i in range(num_classes) if i not in id_split_4]
ood_train_split_4 = random.sample(remaining, num_ood_train)
remaining = [i for i in remaining if i not in ood_train_split_4]
ood_val_split_4 = random.sample(remaining, num_ood_val)
ood_test_split_4 = [i for i in remaining if i not in ood_val_split_4]

id_ood_four_way_split_4 = FourWayClassSplit(id_split_4, ood_train_split_4, ood_val_split_4, ood_test_split_4)

with open("id-ood-splits/split4.pkl", "wb") as f:
   pkl.dump(id_ood_four_way_split_4, f)