import random
import numpy as np
import torch
import random
import glob

from typing import Optional
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


def seed_all(seed):
    # ensure reproducibility
    if not seed:
        seed = 111

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def select_ood_idxs(dataset, imood_class_ids):
    """
    from all imagenet21k classes select those which are ood for imagenet1k from a given list
    """
    img_ids = (
        (torch.tensor(dataset.targets)[..., None] == torch.tensor(imood_class_ids))
        .any(-1)
        .nonzero(as_tuple=True)[0]
    )
    return Subset(dataset, img_ids)


def collate_fn(batch):
    # output formated as : sender input, labels, receiver input
    return (
        torch.stack([x[0][0] for x in batch], dim=0),
        torch.cat([torch.Tensor([x[1]]).long() for x in batch], dim=0),
        torch.stack([x[0][1] for x in batch], dim=0),
    )


def get_dataloader(
    dataset_dir: str,
    dataset_name: str,
    annotations_file: str,
    imood_class_ids: Optional[np.ndarray] = None,
    batch_size: int = 64,
    image_size: int = 384,
    num_workers: int = 4,
    seed: int = 111,
    pin_memory: bool = True,
):
    """
    Parameters
    ----------
    dataset_dir : str
        path to the dataset
    dataset_name : str
        name of the dataset, either 'imagenet_ood' or 'imagenet_val'
    annotations_file : str
        path to the annotations file used to determine the labels
        if devkit downloaded from imagenet website, situated at this location :
        "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
    batch_size : int, optional
        batch size, by default 64
    image_size : int, optional
        size of the images, by default 384 (fits ViT architecture)
    num_workers : int, optional
        number of workers, 4 used in the paper, by default 4
    seed : int, optional
        seed for the random number generator, 111 used in the paper
    pin_memory : bool, optional
        pins memory for faster transfer to GPU, by default True
    """
    # Reset random
    seed_all(seed)

    # Set image resize and normalisation
    transformations = ImageTransformation(image_size)

    # Get dataset
    if dataset_name == "imagenet_ood":
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)
        train_dataset = select_ood_idxs(train_dataset, imood_class_ids)
        clfn_imood = Imood_collate_fn(imood_class_ids).collate_fn_imood
    elif dataset_name == "imagenet_val":
        train_dataset = ImagenetValDataset(
            dataset_dir,
            annotations_file=annotations_file,
            transform=transformations,
        )
    else:
        raise NotImplemented(
            "This Dataset is not available, choose either imagenet_ood or imagenet_val"
        )

    # Split dataset
    test_dataset, train_dataset = torch.utils.data.random_split(
        train_dataset,
        [len(train_dataset) // 10, len(train_dataset) - (len(train_dataset) // 10)],
        torch.Generator().manual_seed(seed),
    )

    # Get dataloaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=clfn_imood if dataset_name == "imagenet_ood" else collate_fn,
        drop_last=True,
        pin_memory=pin_memory,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=clfn_imood if dataset_name == "imagenet_ood" else collate_fn,
        drop_last=True,
        pin_memory=pin_memory,
    )

    return test_loader, train_loader


class ImagenetValDataset(Dataset):
    """
    Custom dataset for imagenet validation set
    Provides labels for each image using the annotations file
    """

    def __init__(self, img_dir, annotations_file, transform=None):
        with open(annotations_file) as f:
            self.img_labels = [int(line) for line in f.readlines()]
        self.transform = transform
        self.files = sorted(glob.glob(f"{img_dir}/*.JPEG"))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = self.pil_loader(img_path)
        label = self.img_labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class ImageTransformation:
    """
    Set of transformations for the image dataset to ensure that the images are of the same size
    and that they are normalized. An additional transformation is applied to the images to deal
    with the fact that the images are in RGB format and the model expects a 3 channel image.
    Parameters
    ----------
    size: int
        The desired size of the images
    """

    def __init__(self, size: int):

        transformations = [
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(transformations)

    def __call__(self, x):
        out_x = self.transform(x)
        return out_x, out_x


class Imood_collate_fn:
    """
    Custom collate function for the iMoOD dataset
    constructed as a class to allow for the use of the class ids
    Parameters
    ----------
    class_ids : np.ndarray
        array of class ids to use as labels
    """

    def __init__(self, imood_class_ids):
        self.imood_class_ids = imood_class_ids

    def collate_fn_imood(batch):
        sender_input = torch.stack([x[0][0] for x in batch], dim=0)
        # labels are corrected for out of domain selection
        labels = (
            torch.cat(
                [
                    torch.Tensor([np.where(self.imood_class_ids == x[1])[0][0]]).long()
                    for x in batch
                ],
                dim=0,
            ),
        )
        receiver_input = torch.stack([x[0][1] for x in batch], dim=0)
        return sender_input, labels, receiver_input
