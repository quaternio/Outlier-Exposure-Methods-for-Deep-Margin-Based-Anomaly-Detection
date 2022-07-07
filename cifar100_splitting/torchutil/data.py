from abc import ABC, abstractmethod

import torch
import torchvision
import PIL
import numpy as np

from xml.etree.ElementTree import parse as ET_parse

class Dataset(torch.utils.data.Dataset, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_target(self, idx):
        raise NotImplementedError
    
    def targets(self):
        for idx in range(len(self)):
            target = self.get_target(idx)
            yield target

class SegmentationDataset(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_mask(self):
        raise NotImplementedError

    def masks(self):
        for idx in range(len(self)):
            mask = self.get_mask(idx)
            yield mask

class ImageFolder(Dataset):
    def __init__(self, root, transform = None, target_transform = None):
        super().__init__()
        self.underlying_dataset = torchvision.datasets.ImageFolder(root, transform = transform, target_transform = target_transform)

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        return self.underlying_dataset[idx]

    def get_target(self, idx):
        target = self.underlying_dataset.targets[idx]
        if self.underlying_dataset.target_transform is not None:
            target = self.underlying_dataset.target_transform(target)
        return target

class Subset(Dataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        inner_idx = self.indices[idx]
        return self.dataset[inner_idx]

    def get_target(self, idx):
        inner_idx = self.indices[idx]
        return self.dataset.get_target(inner_idx)

class ConcatDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
    
    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx -= len(dataset)
        
        raise IndexError

    def get_target(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset.get_target(idx)
            else:
                idx -= len(dataset)
        
        raise IndexError

class CIFAR10(Dataset):
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        super().__init__()
        self.underlying_dataset = torchvision.datasets.CIFAR10(root, train = train, transform = transform, target_transform = target_transform, download = download)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        return self.underlying_dataset[idx]
    
    def get_target(self, idx):
        target = self.underlying_dataset.targets[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target

class CIFAR100(Dataset):
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        super().__init__()
        self.underlying_dataset = torchvision.datasets.CIFAR100(root, train = train, transform = transform, target_transform = target_transform, download = download)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        return self.underlying_dataset[idx]
    
    def get_target(self, idx):
        target = self.underlying_dataset.targets[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target

class SVHN(Dataset):
    def __init__(self, root, split = 'train', transform = None, target_transform = None, download = False):
        super().__init__()
        self.underlying_dataset = torchvision.datasets.SVHN(root, split = split, transform = transform, target_transform = target_transform, download = download)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        return self.underlying_dataset[idx]
    
    def get_target(self, idx):
        target = self.underlying_dataset.targets[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target

class VOCDetection(Dataset):
    def __init__(self, root, year = "2012", image_set = "train", download = False, transform = None, target_transform = None):
        super().__init__()
        self.underlying_dataset = torchvision.datasets.VOCDetection(root, year = year, image_set = image_set, download = download, transform = transform, target_transform = target_transform)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        return self.underlying_dataset[idx]
    
    def get_target(self, idx):
        target = self.underlying_dataset.parse_voc_xml(ET_parse(self.underlying_dataset.annotations[idx]).getroot())
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target

class GenericWrapperDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.underlying_dataset = dataset

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        return self.underlying_dataset[idx]

    def get_target(self, idx):
        item = self.underlying_dataset[idx]
        return item[1]

class LabelMappingDataset(Dataset):
    def __init__(self, dataset, label_list = None, label_mapping = None):
        super().__init__()
        self.dataset = TransformingDataset(dataset, target_transform = LabelMappingTransform(label_list = label_list, label_mapping = label_mapping))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_target(self, idx):
        return self.dataset.get_target(idx)

class LabelMappingTransform:
    def __init__(self, label_list = None, label_mapping = None):
        if label_list is not None:
            label_mapping = {}
            new_label = 0
            for label in label_list:
                label_mapping[label] = new_label
                new_label += 1
            self.label_mapping = label_mapping
        elif label_mapping is not None:
            self.label_mapping = label_mapping
        else:
            raise NotImplementedError

    def __call__(self, label):
        if torch.is_tensor(label):
            label = label.item()
        return self.label_mapping[label]

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        res = self.dataset.__getitem__(index)
        res_list = list(res)
        res_list.append(index)
        return tuple(res_list)

    def get_target(self, idx):
        return self.dataset.get_target(idx)

class TransformingDataset(Dataset):
    def __init__(self, dataset, transform = None, target_transform = None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        res = self.dataset.__getitem__(index)
        res_list = list(res)
        if self.transform is not None:
            res_list[0] = self.transform(res_list[0])
        if self.target_transform is not None:
            res_list[1] = self.target_transform(res_list[1])
        return tuple(res_list)

    def get_target(self, idx):
        target = self.dataset.get_target(idx)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target

class SizeFilteringDataset(Dataset):
    def __init__(self, dataset, min_size):
        super().__init__()
        indices = []
        for idx, (item, _) in enumerate(dataset):
            if item.size[0] >= min_size and item.size[1] >= min_size:
                indices.append(idx)
        
        self.dataset = Subset(dataset, indices)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

    def get_target(self, idx):
        return self.dataset.get_target(idx)

class FixedTargetTransform:
    def __init__(self, target):
        self.target = target

    def __call__(self, prev_target):
        return self.target

class SegmentationMaskToTensorTransform:
    def __call__(self, segmentation_mask):
        # Converts mask to tensor of shape (1, H, W). The extra dimension is
        # needed for resizes and crops.
        return torch.from_numpy(np.array(segmentation_mask)).long().unsqueeze(0)

class SegmentationMaskToLabelSetTransform:
    def __call__(self, segmentation_mask):
        return torch.unique(segmentation_mask)

class LabelSetFilterTransform:
    def __init__(self, remaining_labels):
        self.remaining_labels = remaining_labels
    
    def __call__(self, label_set):
        # Compare each label in label_set to each label in self.remaining_labels,
        # and determine which labels in label_set match at least one label in
        # self.remaining_labels
        label_mask = (label_set.unsqueeze(1) == self.remaining_labels).int().sum(dim = 1) > 0
        # Index the labels in label_set with no matches in self.omitted_labels
        return label_set[label_mask]

class LabelSetToSingleLabelTransform:
    def __call__(self, label_set):
        if len(label_set) != 1:
            raise NotImplementedError
        return label_set[0]

class RelocateTransform:
    def __init__(self, device):
        self.device = device

    def __call__(self, item):
        return item.to(device)

class AnomalyDetectionDataset(Dataset):
    def __init__(self, nominal_dataset, anomaly_dataset):
        super().__init__()
        nominal_target_transform = FixedTargetTransform(0)
        anomaly_target_transform = FixedTargetTransform(1)
        relabeled_nominal_dataset = TransformingDataset(nominal_dataset, target_transform = nominal_target_transform)
        relabeled_anomaly_dataset = TransformingDataset(anomaly_dataset, target_transform = anomaly_target_transform)
        self.dataset = ConcatDataset((relabeled_nominal_dataset, relabeled_anomaly_dataset))
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_target(self, idx):
        return self.dataset.get_target(idx)

# For a given dataset whose targets are label sets, finds the indices of the
# the items whose target label sets include only a single label. Useful for
# converting a segmentation dataset to a classification dataset by reduction.
def get_indices_of_single_labels(dataset):
    indices = []
    for idx, label_set in enumerate(dataset.targets()):
        if len(label_set) == 1:
            indices.append(idx)
    for idx in indices:
        target = dataset.get_target(idx)
    return indices

def get_indices_of_labels(dataset, labels):
    indices = []
    for idx, label in enumerate(dataset.targets()):
        if label in labels:
            indices.append(idx)
    return indices

class SemanticToObjectSegmentationMaskTransform:
    def __init__(self, relevant_labels):
        self.relevant_labels = torch.tensor(relevant_labels, dtype = torch.long)

    def __call__(self, semantic_segmentation_mask):
        # Compare each pixel of the semantic segmentation mask to each relevant
        # label, and determine which pixels have at least (and therefore
        # exactly, assuming relevant labels are unique) one match
        return ((semantic_segmentation_mask.unsqueeze(-1) == self.relevant_labels).int().sum(-1) > 0).int()

class SegmentationLabelMappingTransform:
    def __init__(self, label_map):
        self.label_map = label_map
    
    def __call__(self, mask):
        for src, dst in self.label_map.items():
            mask[mask == src] = dst
        return mask

class SegmentationClassificationDataset(Dataset):
    def __init__(self, segmentation_dataset, remaining_labels):
        super().__init__()
        # Converting a segmentation dataset to a classification dataset involves
        # first converting the segmentation masks to sets of labels which are
        # included in the image, and then removing some labels from those sets
        # which correspond to unimportant or conflicting objects (e.g.
        # "background" and "void" labels, as well as labels of objects which
        # are almost always seen alongside other objects). The goal is to get
        # a set of labels for which most images in the dataset consist of
        # pixels for exactly one of those labels.
        classification_target_transform = torchvision.transforms.Compose([
            SegmentationMaskToLabelSetTransform(), # Convert mask tensor to set of unique labels
            LabelSetFilterTransform(torch.tensor(remaining_labels, dtype = torch.long)) # Remove the filtered labels
        ])
        intermediate_classification_dataset = TransformingDataset(segmentation_dataset, target_transform = classification_target_transform)

        # Next, filter out all images which have more or less than one label
        single_label_indices = get_indices_of_single_labels(intermediate_classification_dataset)
        filtered_dataset = Subset(intermediate_classification_dataset, single_label_indices)
        
        # Labels are still in label sets despite each containing only a single
        # element, i.e. tensors of size one. Reduce them to scalars, and remap
        # the remaining labels to [0, ..., K-1]
        final_target_transform = torchvision.transforms.Compose([
            LabelSetToSingleLabelTransform(), # Reduce size-1 sets to scalars
            LabelMappingTransform(remaining_labels) # Remap remaining labels
        ])
        self.classification_dataset = TransformingDataset(filtered_dataset, target_transform = final_target_transform)
        
        # Now we have a classification dataset, but we still want the 
        # segmentation masks. However, these segmentation masks should be
        # transformed; they should not include the filtered labels, but more
        # accurately, they should be binary masks. Each remaining image has
        # pixels only a single relevant label, the value of which is determined
        # by the classification target. So segmentation masks need to be
        # transformed to 0 / 1, 0 meaning the pixel label is filtered, and 1
        # meaning the pixel label is relevant.

        # First, filter the segmentation_dataset using the same indices as were
        # used to filter the intermediate classification dataset.
        filtered_segmentation_dataset = Subset(segmentation_dataset, single_label_indices)

        # Convert segmentation mask labels to 0/1 as described
        object_segmentation_transform = SemanticToObjectSegmentationMaskTransform(remaining_labels)
        self.object_segmentation_dataset = TransformingDataset(filtered_segmentation_dataset, target_transform = object_segmentation_transform)

    def __len__(self):
        return len(self.classification_dataset)
    
    def __getitem__(self, idx):
        img, label = self.classification_dataset[idx]
        _, object_segmentation_mask = self.object_segmentation_dataset[idx]
        return img, label, object_segmentation_mask

    def get_target(self, idx):
        target = self.classification_dataset.get_target(idx)
        return target

class SegmentationClassificationCroppingDataset(Dataset):
    def __init__(self, dataset, crop, crop_randomly = False):
        super().__init__()
        self.dataset = dataset
        if crop_randomly:
            self.transform = torchvision.transforms.RandomCrop(crop)
        else:
            self.transform = torchvision.transforms.CenterCrop(crop)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, mask = self.dataset[idx]
        merged = torch.cat((img, mask), dim = 0)
        transformed = self.transform(merged)
        transformed_img = transformed[0:3]
        transformed_mask = transformed[3:4].to(torch.long)
        return transformed_img, label, transformed_mask

    def get_target(self, idx):
        return self.dataset.get_target(idx)

class SegmentationClassificationFlippingDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.transform = torchvision.transforms.RandomHorizontalFlip()
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, mask = self.dataset[idx]
        merged = torch.cat((img, mask), dim = 0)
        transformed = self.transform(merged)
        transformed_img = transformed[0:3]
        transformed_mask = transformed[3:4].to(torch.long)
        return transformed_img, label, transformed_mask

    def get_target(self, idx):
        return self.dataset.get_target(idx)

class SegmentationCroppingDataset(Dataset):
    def __init__(self, dataset, crop, crop_randomly = False):
        super().__init__()
        self.dataset = dataset
        if crop_randomly:
            self.transform = torchvision.transforms.RandomCrop(crop)
        else:
            self.transform = torchvision.transforms.CenterCrop(crop)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        merged = torch.cat((img, mask), dim = 0)
        transformed = self.transform(merged)
        transformed_img = transformed[0:3]
        transformed_mask = transformed[3:4].to(torch.long)
        return transformed_img, transformed_mask

    def get_target(self, idx):
        return self.dataset.get_target(idx)

class SegmentationFlippingDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.transform = torchvision.transforms.RandomHorizontalFlip()
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        merged = torch.cat((img, mask), dim = 0)
        transformed = self.transform(merged)
        transformed_img = transformed[0:3]
        transformed_mask = transformed[3:4].to(torch.long)
        return transformed_img, transformed_mask

    def get_target(self, idx):
        return self.dataset.get_target(idx)

class VOCDetectionClassificationDataset(Dataset):
    def __init__(self, detection_dataset, object_name_mapping):
        super().__init__()
        self.detection_dataset = detection_dataset
        self.object_name_mapping = object_name_mapping

    def __len__(self):
        return len(self.detection_dataset)

    def __getitem__(self, idx):
        image, target = self.detection_dataset[idx]
        target = [self.object_name_mapping[o['name']] for o in target['annotation']['object'] if o['name'] in self.object_name_mapping]
        target = torch.tensor(target, dtype = torch.long)
        target = torch.unique(target)
        return image, target

    def get_target(self, idx):
        detection_target = self.detection_dataset.get_target(idx)
        target = [self.object_name_mapping[o['name']] for o in detection_target['annotation']['object'] if o['name'] in self.object_name_mapping]
        target = torch.tensor(target, dtype = torch.long)
        target = torch.unique(target)
        return target

class SegmentationClassificationToSegmentationDataset(Dataset, SegmentationDataset):
    def __init__(self, segmentation_classification_dataset):
        self.dataset = segmentation_classification_dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, mask = self.dataset[idx]
        target = mask * (label + 1)
        return image, target

    def get_target(self, idx):
        return self.dataset.get_target(idx)

    def get_mask(self, idx):
        _, label, mask = self.dataset[idx]
        target = mask * (label + 1)
        return target

# Since this adds random augmentations, it cannot return masks
# deterministically, so it does not extend SegmentationDataset
class CopyPasteAugmentationSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, segmentation_dataset, n_copy_paste, ignore_labels = None, degrees = 0, translate = None, scale = None, shear = None):
        self.dataset = segmentation_dataset
        self.n_copy_paste = n_copy_paste
        if ignore_labels is None:
            self.ignore_labels = None
        else:
            self.ignore_labels = torch.tensor(ignore_labels, dtype = torch.long)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        base_image, base_mask = self.dataset[idx]
        for _ in range(self.n_copy_paste):
            # Get another random image
            cur_random_idx = np.random.randint(len(self.dataset))
            cur_random_paste_image, cur_random_paste_mask = self.dataset[cur_random_idx]
            
            # Get the list of unique labels from the random image
            cur_unique_pixel_labels = torch.unique(cur_random_paste_mask)

            # Remove any of the labels which are to be ignored in pasting
            if self.ignore_labels is not None:
                cur_retained_unique_pixel_label_indices = (cur_unique_pixel_labels.unsqueeze(-1) == self.ignore_labels).to(torch.int).sum(dim = -1) == 0
                cur_unique_pixel_labels = cur_unique_pixel_labels[cur_retained_unique_pixel_label_indices]
            
            # If there are no random unique pixel labels remaining,
            # simply continue
            if len(cur_unique_pixel_labels) == 0:
                continue

            # Choose a random unique pixel label
            cur_random_pixel_label = cur_unique_pixel_labels[torch.randint(len(cur_unique_pixel_labels), (1,))[0]]
            
            # Select the part of the mask corresponding to that label (i.e.
            # convert to binary mask)
            cur_random_matching_mask = (cur_random_paste_mask == cur_random_pixel_label)

            # Apply random affine transform to the random image and mask with 
            # NEAREST interpolation for maintaining binary mask. 
            random_degrees = float(torch.rand(1)[0] - 0.5 * 2 * self.degrees)
            
            if self.translate is None:
                random_translate = [0, 0]
            else:
                horizontal_end = int(cur_random_matching_mask.shape[2] * self.translate[0])
                vertical_end = int(cur_random_matching_mask.shape[1] * self.translate[1])
                random_horizontal_translate = int(torch.randint(2 * horizontal_end, (1,))[0] - horizontal_end)
                random_vertical_translate = int(torch.randint(2 * vertical_end, (1,))[0] - vertical_end)
                random_translate = [random_horizontal_translate, random_vertical_translate]

            if self.scale is None:
                random_scale = 1.0
            else:
                random_scale = float(torch.rand(1)[0] * (self.scale[1] - self.scale[0]) + self.scale[0])

            if self.shear is None:
                random_shear = 0.0
            elif type(self.shear) is list:
                random_horizontal_shear = float(torch.rand(1)[0] - 0.5 * 2 * self.shear[0])
                random_vertical_shear = float(torch.rand(1)[0] - 0.5 * 2 * self.shear[1])
                random_shear = [random_horizontal_shear, random_vertical_shear]
            else:
                random_shear = float(torch.rand(1)[0] - 0.5 * 2 * self.shear)
            
            cur_transformed_random_paste_image = torchvision.transforms.functional.affine(cur_random_paste_image, angle = random_degrees, translate = random_translate, scale = random_scale, shear = random_shear, interpolation = torchvision.transforms.InterpolationMode.BILINEAR)
            cur_transformed_random_matching_mask = torchvision.transforms.functional.affine(cur_random_matching_mask.to(torch.float), angle = random_degrees, translate = random_translate, scale = random_scale, shear = random_shear, interpolation = torchvision.transforms.InterpolationMode.NEAREST, fill = 0)
            
            # Copy / paste the image data via a 0/1 mixing between the base
            # image and the cur_random_paste_image, using the binary mask as
            # coefficients
            base_image = cur_transformed_random_matching_mask * cur_transformed_random_paste_image + (1 - cur_transformed_random_matching_mask) * base_image

            # Use the binary mask to index the current base image
            # mask and assign the labels to the randomly chosen unique pixel
            # label
            base_mask[cur_transformed_random_matching_mask.to(torch.bool)] = cur_random_pixel_label

        # Return the updated base image and base mask
        return base_image, base_mask

class MemoryCachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cached_examples = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx in self.cached_examples:
            return self.cached_examples[idx]
        else:
            example = self.dataset[idx]
            self.cached_examples[idx] = example
            return example

def bipartition_dataset(dataset, p = 0.5):
    full_indices = list(range(len(dataset)))

    num = int(len(dataset) * p)
    first_indices = list(range(num))
    first_indices = [int(float(index) / p) for index in first_indices]
    
    #second_indices = full_indices
    #for i in range(len(first_indices)):
    #    idx = len(second_indices) - i - 1
    #    inner_idx = first_indices[idx]
    #    del second_indices[inner_idx]
    first_indices_set = set(first_indices)
    second_indices = [index for index in full_indices if index not in first_indices_set]

    first_dataset = Subset(dataset, first_indices)
    second_dataset = Subset(dataset, second_indices)
    return first_dataset, second_dataset

class_counts = {
    'cifar10': 10,
    'cifar100': 100,
    'svhn': 10,
    'imagenette': 10,
    'boeing': 2,
    'boeing-out': 1,
    'voc_segmentation_classification': 18, # 18 remaining after removing 2
    'ilsvrc_2012_intersecting_voc_segmentation_classification': 15,
    'voc_detection_classification': 18, # 18 remaining after removing 2
    'ilsvrc_2012_strengthened_voc_detection_classification': 15,
    'imagenet1k': 1000,
    'imagenet_osr_easy': 1000,
    'imagenet_osr_hard': 1000,
    'voc_segmentation_2012': 21,
    'ilsvrc_2012_intersecting_voc_segmentation_all': 16
}

def add_resize_crop_transforms(transform, resize = None, crop = None, crop_randomly = False):
    result = []
    if transform is not None:
        result.append(transform)
    
    if resize is not None:
        result.append(torchvision.transforms.Resize(resize))
    
    if crop is not None:
        if crop_randomly:
            result.append(torchvision.transforms.RandomCrop(crop))
        else:
            result.append(torchvision.transforms.CenterCrop(crop))

    return torchvision.transforms.Compose(result)

def _get_voc_detection_classification_dataset(root_dir, object_name_mapping, split = 'train', transform = None, resize = None, crop = None, crop_randomly = False, download = True):
    # Decide on data split
    if split == 'train':
        image_set = 'train'
    elif split == 'test':
        image_set = 'val'
    else:
        return NotImplemented

    transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
    detection_dataset = VOCDetection(root_dir, image_set = image_set, transform = transform, download = download)
    multilabel_classification_dataset = VOCDetectionClassificationDataset(detection_dataset, object_name_mapping)
    single_label_indices = get_indices_of_single_labels(multilabel_classification_dataset)
    torch.save(single_label_indices, 'single_label_indices.pth')
    filtered_multilabel_classification_dataset = Subset(multilabel_classification_dataset, single_label_indices)
    final_target_transform = LabelSetToSingleLabelTransform()
    classification_dataset = TransformingDataset(filtered_multilabel_classification_dataset, target_transform = LabelSetToSingleLabelTransform())
    return classification_dataset

def _get_voc_segmentation_classification_dataset(root_dir, remaining_labels, split = 'train', transform = None, resize = None, crop = None, crop_randomly = False, download = True, target_transform = None, flip_randomly = False):
    # Resizing and cropping are different for segmentation datasets.
    # The mask must also be resized and cropped using the same coordinates
    # as the image, but the resized mask must be interpolated differently to
    # prevent mixing of labels. To handle consistency in random crops, the
    # resized image and mask will be merged into a single (C+1)xHxW tensor,
    # cropped once, and then separated, all via a
    # SegmentationClassificationCroppingDataset.
    
    # First, get dataset, applying resize and other transforms
    transform = add_resize_crop_transforms(transform, resize)
    target_transform_list = [SegmentationMaskToTensorTransform()]
    if target_transform is not None:
        target_transform_list.append(target_transform)
    if resize is not None:
        target_transform_list.append(torchvision.transforms.Resize(resize, interpolation = torchvision.transforms.InterpolationMode.NEAREST))
    target_transform = torchvision.transforms.Compose(target_transform_list)

    # Decide on data split
    if split == 'train':
        image_set = 'train'
    elif split == 'test':
        image_set = 'val'
    else:
        return NotImplemented

    raw_dataset = GenericWrapperDataset(torchvision.datasets.VOCSegmentation(root_dir, image_set = image_set, transform = transform, target_transform = target_transform, download = download))

    segmentation_classification_dataset = SegmentationClassificationDataset(raw_dataset, remaining_labels)
    
    # Next, the image and mask need to be cropped consistently. This is
    # done with a SegmentationClassificationCroppingDataset.
    if crop is not None:
        cropped_dataset = SegmentationClassificationCroppingDataset(segmentation_classification_dataset, crop, crop_randomly)
    else:
        cropped_dataset = segmentation_classification_dataset

    if flip_randomly:
        flipped_dataset = SegmentationClassificationFlippingDataset(cropped_dataset)
    else:
        flipped_dataset = cropped_dataset
    
    return flipped_dataset

def get_segmentation_class_balanced_loss_weights(dataset):
    counts = {}
    greatest_label = -1
    for mask in dataset.masks():
        cur_label_set = torch.unique(mask)
        for label in cur_label_set:
            label = int(label.item())
            greatest_label = label if label > greatest_label else greatest_label
            cur_count = int((mask == label).to(torch.int).sum().item())
            if label in counts:
                counts[label] += cur_count
            else:
                counts[label] = cur_count
    
    counts_list = [val for _, val in counts.items()]
    counts_list.sort()
    if len(counts_list) % 2 == 1:
        median_count = counts_list[len(counts_list) // 2]
    else:
        right = len(counts_list) // 2
        left = right - 1
        median_count = (float(counts_list[left]) + float(counts_list[right])) / 2.0
    
    weights = []
    for label in range(greatest_label + 1):
        if label in counts:
            weights.append(median_count / counts[label])
        else:
            weights.append(0.0)
    
    return torch.tensor(weights)

def get_class_balanced_sampler_weights(dataset):
    counts = {}
    labels = []
    for label in dataset.targets():
        if torch.is_tensor(label):
            label = label.item()
        labels.append(label)
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    
    label_weights = {}
    
    for label in counts:
        label_weights[label] = 1.0 / counts[label]

    weights = [label_weights[label] for label in labels]
    
    return torch.tensor(weights)

def get_dataset(dataset_name, root_dir, split = 'train', transform = None, resize = None, crop = None, crop_randomly = False, download = True, target_transform = None, flip_randomly = False):
    # Make sure image gets converted to tensor
    if transform is not None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            transform
        ])
    else:
        transform = torchvision.transforms.ToTensor()

    if dataset_name == 'cifar10':
        transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
        return CIFAR10(root_dir, train = (split == 'train'), transform = transform, download = download)
    elif dataset_name == 'cifar100':
        transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
        return CIFAR100(root_dir, train = (split == 'train'), transform = transform, download = download)
    elif dataset_name == 'svhn':
        transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
        return SVHN(root_dir, split = split, transform = transform, download = download)
    elif dataset_name == 'imagenette':
        transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
        return ImageFolder(f'{root_dir}/imagenette/{split}', transform = transform)
    elif dataset_name == 'boeing':
        transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
        return ImageFolder(f'{root_dir}/boeing/{split}', transform = transform)
    elif dataset_name == 'boeing-out':
        transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
        return ImageFolder(f'{root_dir}/boeing/{split}-out', transform = transform)
    elif dataset_name == 'voc_segmentation_classification':
        # Background, void, and two other labels rarely seen on their own
        omitted_labels = [0, 255, 15, 9]
        remaining_labels = [x for x in range(1, 21) if x not in set(omitted_labels)]
        return _get_voc_segmentation_classification_dataset(root_dir, remaining_labels, split = split, transform = transform, resize = resize, crop = crop, crop_randomly = crop_randomly, download = download, target_transform = target_transform, flip_randomly = flip_randomly)
    elif dataset_name == 'ilsvrc_2012_intersecting_voc_segmentation_classification':
        # Background, void, two other labels rarely seen on their own (chair,
        # person), and the other labels which don't appear in imagenet (cow,
        # horse, sheep)
        omitted_labels = [0, 255, 15, 9, 10, 13, 17]
        remaining_labels = [x for x in range(1, 21) if x not in set(omitted_labels)]
        return _get_voc_segmentation_classification_dataset(root_dir, remaining_labels, split = split, transform = transform, resize = resize, crop = crop, crop_randomly = crop_randomly, download = download, target_transform = target_transform, flip_randomly = flip_randomly)
    elif dataset_name == 'voc_detection_classification':
        object_name_mapping = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'cow': 8,
            'diningtable': 9,
            'dog': 10,
            'horse': 11,
            'motorbike': 12,
            'pottedplant': 13,
            'sheep': 14,
            'sofa': 15,
            'train': 16,
            'tvmonitor': 17
        }
        return _get_voc_detection_classification_dataset(root_dir, object_name_mapping, split = split, transform = transform, resize = resize, crop = crop, crop_randomly = crop_randomly, download = download)
    elif dataset_name == 'ilsvrc_2012_strengthened_voc_detection_classification':
        if download:
            return NotImplemented

        ilsvrc_2012_voc_folder = f'{root_dir}/ILSVRC_2012_VOC/{split}'
        ilsvrc_2012_voc_transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
        ilsvrc_2012_voc_dataset = ImageFolder(ilsvrc_2012_voc_folder, transform = ilsvrc_2012_voc_transform)

        object_name_mapping = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'diningtable': 8,
            'dog': 9,
            'motorbike': 10,
            'pottedplant': 11,
            'sofa': 12,
            'train': 13,
            'tvmonitor': 14
        }
        
        voc_detection_classification_dataset = _get_voc_detection_classification_dataset(root_dir, object_name_mapping, split = split, transform = transform, resize = resize, crop = crop, crop_randomly = crop_randomly, download = download)
        torch.save(len(ilsvrc_2012_voc_dataset), 'ilsvrc_len.pth')
        torch.save(len(voc_detection_classification_dataset), 'voc_len.pth')
        
        return ConcatDataset((ilsvrc_2012_voc_dataset, voc_detection_classification_dataset))
    elif dataset_name == 'imagenet1k':
        # The training set isn't actually used here since all the large scale
        # benchmarks use pretrained imagenet models. However, the 'val' split
        # is used to tune density estimation for mahalanobis and similar
        # methods. The 'test' split is used for computing AUCs against an
        # OOD dataset, particularly one of the 'imagenet_osr_{easy|hard}'
        # datasets, as in Vaze et al.
        
        # In-distribution labels aren't necessary to compute AUC, so it assumes
        # that all of the ImageNet validation images are located in one root
        # folder within <root_dir>/imagenet1k, such as 
        # <root_dir>/imagenet1k/val. This is similar to the default file
        # structure of the ILSVRC2012 validation data.
        ilsvrc_2012_folder = f'{root_dir}/imagenet1k/{split}'
        transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
        ilsvrc_2012_dataset = ImageFolder(ilsvrc_2012_folder, transform = transform)
        return ilsvrc_2012_dataset
    elif dataset_name == 'imagenet_osr_easy':
        # Load the "easy" OOD large scale ImageNet dataset. Comes from
        # ImageNet-21K-P, so the data is naturally structured in train/val
        # splits, with 50 validation images in each class. However, since these
        # are used for OOD data (and thus not seen during training), they will
        # likely be concatenated together in the end. Nonetheless, they are
        # still separated here into train and val. The 'test' split is
        # remapped to 'val' in name
        if split == 'test':
            split = 'val'
        imagenet_osr_easy_folder = f'{root_dir}/imagenet_osr_easy/{split}'
        transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
        imagenet_osr_easy_dataset = ImageFolder(imagenet_osr_easy_folder, transform = transform)
        return imagenet_osr_easy_dataset
    elif dataset_name == 'imagenet_osr_hard':
        # Load the "hard" OOD large scale ImageNet dataset. Comes from
        # ImageNet-21K-P, so the data is naturally structured in train/val
        # splits, with 50 validation images in each class. However, since these
        # are used for OOD data (and thus not seen during training), they will
        # likely be concatenated together in the end. Nonetheless, they are
        # still separated here into train and val. The 'test' split is
        # remapped to 'val' in name
        if split == 'test':
            split = 'val'
        imagenet_osr_hard_folder = f'{root_dir}/imagenet_osr_hard/{split}'
        transform = add_resize_crop_transforms(transform, resize, crop, crop_randomly)
        imagenet_osr_hard_dataset = ImageFolder(imagenet_osr_hard_folder, transform = transform)
        return imagenet_osr_hard_dataset
    elif dataset_name == 'voc_segmentation_2012':
        transform = add_resize_crop_transforms(transform, resize)
        target_transform_list = [SegmentationMaskToTensorTransform()]
        if target_transform is not None:
            target_transform_list.append(target_transform)
        if resize is not None:
            target_transform_list.append(torchvision.transforms.Resize(resize, interpolation = torchvision.transforms.InterpolationMode.NEAREST))
        label_map = {
            255: 0
        }
        target_transform_list.append(SegmentationLabelMappingTransform(label_map))
        target_transform = torchvision.transforms.Compose(target_transform_list)
        
        # Decide on data split
        if split == 'train':
            image_set = 'train'
        elif split == 'test':
            image_set = 'val'
        else:
            return NotImplemented
        
        raw_dataset = GenericWrapperDataset(torchvision.datasets.VOCSegmentation(root_dir, image_set = image_set, transform = transform, target_transform = target_transform, download = download))
        if crop is not None:
            cropped_dataset = SegmentationCroppingDataset(raw_dataset, crop, crop_randomly)
        else:
            cropped_dataset = raw_dataset
        
        if flip_randomly:
            flipped_dataset = SegmentationFlippingDataset(cropped_dataset)
        else:
            flipped_dataset = cropped_dataset
        
        return flipped_dataset
    elif dataset_name == 'ilsvrc_2012_intersecting_voc_segmentation_all':
        transform = add_resize_crop_transforms(transform, resize)
        target_transform_list = [SegmentationMaskToTensorTransform()]
        if target_transform is not None:
            target_transform_list.append(target_transform)
        if resize is not None:
            target_transform_list.append(torchvision.transforms.Resize(resize, interpolation = torchvision.transforms.InterpolationMode.NEAREST))
        label_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 0,
            10: 0,
            11: 9,
            12: 10,
            13: 0,
            14: 11,
            15: 0,
            16: 12,
            17: 0,
            18: 13,
            19: 14,
            20: 15,
            255: 0
        }
        target_transform_list.append(SegmentationLabelMappingTransform(label_map))
        target_transform = torchvision.transforms.Compose(target_transform_list)
        
        # Decide on data split
        if split == 'train':
            image_set = 'train'
        elif split == 'test':
            image_set = 'val'
        else:
            return NotImplemented
        
        raw_dataset = GenericWrapperDataset(torchvision.datasets.VOCSegmentation(root_dir, image_set = image_set, transform = transform, target_transform = target_transform, download = download))
        if crop is not None:
            cropped_dataset = SegmentationCroppingDataset(raw_dataset, crop, crop_randomly)
        else:
            cropped_dataset = raw_dataset
        
        if flip_randomly:
            flipped_dataset = SegmentationFlippingDataset(cropped_dataset)
        else:
            flipped_dataset = cropped_dataset
        
        return flipped_dataset
