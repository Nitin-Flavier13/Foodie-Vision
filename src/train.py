
import torch 
from torch import nn 
from torchvision import transforms
from typing import List,Dict,Tuple

import engine, utils, transform
from models.TinyVGG import TinyVGGV1
from data_setup import create_dataloaders
from timeit import default_timer as timer

NUM_EPOCHS=1
BATCH_SIZE=25
NUM_WORKERS=1
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
torch.cuda.manual_seed(42)

train_dir = f'../dataset/MiniFood101/train'
test_dir = f'../dataset/MiniFood101/test'

device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transform.train_transform1
test_transform = transform.test_transform1

if __name__ == "__main__":
    
    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                        test_dir=test_dir,
                                                                        train_transform=train_transform,
                                                                        test_transform=test_transform,
                                                                        batch_size=BATCH_SIZE,
                                                                        num_workers=NUM_WORKERS)

    model = TinyVGGV1(input_shape=3,
                    hidden_units=10,
                    output_shape=len(class_names)).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)

    start_time = timer()
    model_results = engine.initiate_training(model=model,
                                            train_dataloader=train_dataloader,
                                            test_dataloader=test_dataloader,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            # scheduler=scheduler1,
                                            epochs=NUM_EPOCHS,
                                            device=device)       
    end_time = timer()

    model_training_time = end_time-start_time

    model_target_dir = '../models'
    utils.save_model(model=model,
                    target_dir=model_target_dir,
                    model_name='TinyVGGV1.pth')

    utils.upload_performance(model_name='DUMMY_TEST_TinyVGGV1',
                        target_dir='../model_performance/',
                        training_time= model_training_time,
                        model_results=model_results)

