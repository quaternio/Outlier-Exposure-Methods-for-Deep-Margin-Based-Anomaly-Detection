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
import pickle as pkl

import torchutil

with open('id-ood-splits/split1.pkl', 'rb') as f:
    id_ood_split = pkl.load(f) # Type is adtools.data.ClassSplit

print(f'ID classes: {id_ood_split.id_labels()}')
print(f'OOD classes: {id_ood_split.ood_labels()}')

# Wrapper around torchvision.datasets.CIFAR100 which provides an interface for
# getting image labels without loading images from disk (significantly speeds
# up ID/OOD splitting process)
train_dataset = torchutil.data.CIFAR100(root = '/scratch/guyera/anomaly_detection/datasets', train = True, download = True)
test_dataset = torchutil.data.CIFAR100(root = '/scratch/guyera/anomaly_detection/datasets', train = False, download = True)

id_train_dataset, ood_train_dataset = id_ood_split.split_dataset(train_dataset)
id_test_dataset, ood_test_dataset = id_ood_split.split_dataset(test_dataset)

# Map ID labels and OOD labels to [0, 1, 2, ...]
id_train_dataset = torchutil.data.TransformingDataset(
    id_train_dataset,
    target_transform = torchutil.data.LabelMappingTransform(
        label_list = id_ood_split.id_labels()
    )
)
ood_train_dataset = torchutil.data.TransformingDataset(
    ood_train_dataset,
    target_transform = torchutil.data.LabelMappingTransform(
        label_list = id_ood_split.ood_labels()
    )
)
id_test_dataset = torchutil.data.TransformingDataset(
    id_test_dataset,
    target_transform = torchutil.data.LabelMappingTransform(
        label_list = id_ood_split.id_labels()
    )
)
ood_test_dataset = torchutil.data.TransformingDataset(
    ood_test_dataset,
    target_transform = torchutil.data.LabelMappingTransform(
        label_list = id_ood_split.ood_labels()
    )
)

# Get indices of "training classes", which is a subset of the 90 ID classes,
# and a superset of the "evaluation classes"
num_training_classes = 90 # In this example, we use all 90 ID classes for training
training_class_indices = list(range(num_training_classes))

# Get indices of "evaluation classes", which is a subset of the training
# classes. Basically, the evaluation classes are our true ID classes, and the
# training classes are a mix of ID classes and other "auxiliary" classes.
num_evaluation_classes = 10 # These are your ID classes
evaluation_class_indices = list(range(num_evaluation_classes))

# Filter the ID data to only include the training classes. They're already
# in the range [0, 1, ...], so no label remapping is necessary.
id_train_dataset = torchutil.data.Subset(id_train_dataset, torchutil.data.get_indices_of_labels(id_train_dataset, training_class_indices))

# Evaluate only on the evaluation classes.
id_test_dataset = torchutil.data.Subset(id_test_dataset, torchutil.data.get_indices_of_labels(id_test_dataset, evaluation_class_indices))

'''
You may want to split the train dataset into train / val before filtering it
to the training classes. Then you can separately filter it to the training
classes, and filter the val set to the evaluation classes (unless you want to
perform model selection based on the classification performance on all of
the training classes; that's an important design decision)

Of course, you'll likely want to modify your anomaly detector to understand
the difference between training classes and evaluation classes. Training classes
are a union of evaluation classes and auxiliary classes. When you're doing
anomaly detection, you usually only want to look at the evaluation classes.
In the case of max logit, I usually do the following in the score() function
of my anomaly detector class:

1. Make the evaluation_class_indices an argument to the function
2. Compute `logits = self.model(batch)`
3. Filter the logits to only include the logits of the evaluation classes: `logits = logits[:, evaluation_class_indices]`
4. Compute the max filtered logit: `max_logits, _ = torch.max(logits, dim = 1)`
5. The score is the negative max logit: `score = -max_logits`

The strategy to incorporate auxiliary classes will probably depend on the nature
of the anomaly detection method, but I think in general it will look something
like that.
'''