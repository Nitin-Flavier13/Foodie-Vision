''' 
Contains functionality for creating Pytorch DataLoader, for image classification
'''

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       train_transform: transforms.Compose,
                       test_transform: transforms.Compose,
                       batch_size:int,
                       num_workers: int = 1):
    ''' 
    Creates training and testing DataLoaders
    
    Takes in training and testing directory paths and turns 
    them into PyTorch Datasets and into Pytorch DataLoaders

    Args:
        train_dir: Path to Training Directory.
        test_dir: Path to Testing Directory.
        tranform: torchvision tranform to perform 
                  tranformation on train and test dataset.
        batch_size: Number of samples per batch in each of 
                    the DataLoaders. 
        num_workers: number of logical processors
                     per DataLoader.
    
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
    '''

    train_data = datasets.ImageFolder(train_dir,
                                        transform=train_transform)
    test_data = datasets.ImageFolder(test_dir,
                                        transform=test_transform)
    
    class_names = train_data.classes

    train_dataloader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True)   # pin_memory enables fast data transfer to CUDA-enabled GPUs

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=True)

    return train_dataloader,test_dataloader, class_names
