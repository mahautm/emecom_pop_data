# Script to recreate the datasets used in the paper
# There are two datasets:
#   imagenet_ood : The out of domain is created from the imagenet21k dataset by selecting the classes which are not in imagenet1k
#   imagenet_val : a simple split keeping 10% of the imagenet1k validation data for testing
import numpy as np
from data_utils import get_dataloader

if __name__ == "__main__":
    dataset_name = "imagenet_ood"  # "imagenet_ood" or "imagenet_val"
    imood_class_ids = np.load("./imood_class_ids.npy")

    test_loader, train_loader = get_dataloader(
        dataset_name=dataset_name,
        imood_class_ids=imood_class_ids,
        dataset_dir="path_to_datasets/imagenet21k_resized/imagenet21k_train/",
        # annotation file is only required input for imagenet_val, will be ignored otherwise
        annotations_file="path_to_devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
    )

    for batch_idx, batch in enumerate(train_loader):
        sender_input, labels, receiver_input = batch
        raise NotImplementedError("...")
