''' 
Contains function for training and testing a PyTorch Model.
'''
import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    accuracy = (correct/len(y_pred)) * 100
    return accuracy

def training_step(model: nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  loss_fn: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  epoch: int,
                  device: torch.device = device,) -> list[int]:
    ''' 
        Performs training step on the model.
    '''
    
    acc_train = 0
    loss_train = 0
    model.train()

    for X,y in tqdm(data_loader, desc=f"Epoch {epoch} Training", leave=False):
        X,y = X.to(device),y.to(device)
        
        y_pred_logits = model(X)
        y_pred = torch.argmax(y_pred_logits,dim=1)

        loss = loss_fn(y_pred_logits,y)
        loss_train += loss.item()
        acc_train += accuracy_fn(y,y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss_train,acc_train


def testing_step(model: nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  loss_fn: nn.Module,
                  epoch: int,
                  device: torch.device = device,) -> list[int]:
    ''' 
        Performs testing step on the model.
    '''
    acc_test = 0
    loss_test = 0

    model.eval()

    with torch.inference_mode():
        for X,y in tqdm(data_loader, desc=f"Epoch {epoch} Testing", leave=False):
            X,y = X.to(device),y.to(device)

            y_pred_logits = model(X)
            y_pred = torch.argmax(y_pred_logits,dim=1)

            loss_test += loss_fn(y_pred_logits,y).item()
            acc_test += accuracy_fn(y,y_pred)
    
    return loss_test,acc_test


from typing import Dict,List

def initiate_training(model: nn.Module,
                      train_dataloader: torch.utils.data.DataLoader,
                      test_dataloader: torch.utils.data.DataLoader,
                      optimizer: torch.optim.Optimizer,
                      loss_fn: nn.Module,
                    #   scheduler: torch.optim.lr_scheduler._LRScheduler | None=None,
                      epochs: int=5,
                      device=device) -> Dict[str,list[int]]:
    ''' 
    Trains and Tests a PyTorch Model
    Trains and Tests model for each epoch.
    Calculate, print and store evaluation metrics throughout.

    Args:
        model: A pytorch model to be trained and tested.
        train_dataloader: A dataloader instance for the model to be trained on.
        test_dataloader: A dataloader instance for the model to be tested on.
        optimizer: A pytorch optimizer to update weights to reduce loss.
        loss_fn: A PyTorch loss function to evaluate loss.
        epochs: The number of times the entire dataset is passed through the 
                neural network during training.
        device: A target device to compute on ("cpu","cuda")
    
    Returns:
        A dictionary of train/test loss/acc. Each metric has a list containing 
        values for each epoch.
         
    '''
    
    results = {
                "train_loss": [],
                "test_loss": [],
                "train_acc": [],
                "test_acc": [],
                "epoch_count": [i for i in range(1, epochs+1)]
               }
    
    for epoch in range(epochs):
        # Training
        loss_train,acc_train = training_step(model=model,
                                            data_loader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            epoch=epoch,
                                            device=device) 
        
        epoch_train_loss = loss_train / len(train_dataloader)
        epoch_train_acc = acc_train / len(train_dataloader)
        results['train_loss'].append(epoch_train_loss)
        results['train_acc'].append(epoch_train_acc)

        # Testing
        loss_test,acc_test = testing_step(model=model,
                                        data_loader=test_dataloader,
                                        loss_fn=loss_fn,
                                        epoch=epoch,
                                        device=device)
            
        epoch_test_loss = loss_test / len(test_dataloader)
        epoch_test_acc = acc_test / len(test_dataloader)
        results['test_loss'].append(epoch_test_loss)
        results['test_acc'].append(epoch_test_acc)
        
        # scheduler.step(epoch_test_loss)
        # Print final metrics for the epoch
        print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f} | Train Accuracy: {epoch_train_acc:.4f} | Test Accuracy: {epoch_test_acc:.4f}")

    return results

    
