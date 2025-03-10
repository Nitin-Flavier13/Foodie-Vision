'''
- 
-
- create_writer():
    Creates an instance for SummaryWriter():

    - Experiment date
    - Experiment name
    - Model name 

    logging directory: 'runs/YYYY-MM-DD/experiment_name/model_name/extra'
'''


import os
import torch
import pandas as pd

from pathlib import Path
from typing import List,Dict
from datetime import datetime
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    ''' 
    Saves a PyTorch model to the target directory.
    Args:
        model: the model to be saved.
        target_dir: directory to save the model.
        model_name: A filename for the saved model, should include 
                    either ".pth" or ".pt" as their file_extension.
        
    '''
    # create target directory 
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth' " 
    model_save_path = target_dir_path / model_name

    torch.save(obj=model.state_dict(),
               f=model_save_path)

def upload_performance(model_name: str,
                       target_dir: str,
                       batch_size: int,
                       learning_rate: float,
                       training_time: float,
                       model_results: Dict[str,List[int]]):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_training_time = str(timedelta(seconds=training_time))

    model_data = {
        "timestamp": timestamp,
        "model_name": model_name,
        "training_time": formatted_training_time,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": len(model_results["test_acc"]),
        "test_accuracy": model_results["test_acc"][-1],
        "train_accuracy": model_results["train_acc"][-1],
    }
    
    # Convert to a DataFrame
    df = pd.DataFrame([model_data])

    # Append to CSV if it exists, otherwise create a new one
    target_path = target_dir + "performance_comparision.csv"
    
    if os.path.exists(target_dir):
        df.to_csv(target_path, mode='a', header=False, index=False)
    else:
        os.makedirs(target_dir,exist_ok=True)
        df.to_csv(target_path, mode='w', header=True, index=False)

    print("Model results saved successfully! 🚀")

def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None):
    """ 
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instances tracking to a specific experiment
    """

    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join("../runs",timestamp,experiment_name,model_name,extra)
    else:
        log_dir = os.path.join("../runs",timestamp,experiment_name,model_name)
    
    return SummaryWriter(log_dir=log_dir)
    



