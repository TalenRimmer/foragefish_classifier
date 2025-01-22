'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
from tqdm import trange

import torch # this imports pytorch
import torch.nn as nn # this contains our loss function 
from torch.utils.data import DataLoader # the pytorch dataloader class will take care of all kind of parallelization during training
from torch.optim import SGD # this imports the optimizer

# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18
import datetime

#Wandb imports:
import wandb
import random

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['num_workers']
        )
    # print("PUT BACK IN THE BELOW")
    # dataLoader = DataLoader(
    #         dataset=dataset_instance,
    #         batch_size=cfg['batch_size'],
    #         shuffle=split == 'train',
    #         num_workers=cfg['num_workers']
    #     )
    return dataLoader



def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob('model_states/*.pt')

    if cfg['resume_path'] != '':
        checkpoint = torch.load(cfg['resume_path'])
        model_instance.load_state_dict(checkpoint['model'])
        print(f"Resuming the model from {cfg['resume_path']}")

    # if len(model_states):
    #     # at least one save state found; get latest
    #     model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states]
    #     start_epoch = max(model_epochs)

    #     # load state dict and apply weights to model
    #     print(f'Resuming from epoch {start_epoch}')
    #     state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
    #     model_instance.load_state_dict(state['model'])

    # else:
    #     # no save state found; start anew
    #     print('Starting new model')
    start_epoch = 0

    return model_instance, start_epoch



def save_model(cfg, epoch, model, stats, current_val_loss, best_val_loss):
    # make sure save directory exists; create if not
    os.makedirs('model_states', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save only if current val loss is less than the previous 'best' val loss
    if current_val_loss < best_val_loss:
        torch.save(stats, open(f'model_states/best.pt', 'wb')) # Open is opening the file path we specify, and 'wb' is specifying it's in a writable binary file
        best_val_loss = current_val_loss
        print(f"saved the new best model, current_val_loss: {current_val_loss}, current_epoch: {epoch}")

    # torch.save(stats, open(f'model_states/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = 'model_states/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)
    
    return best_val_loss
    


            

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''
    if cfg["learning_rate"] == 0:
        print("MAH WARNING THE LEARNING RATE IS SET TO 0")

    device = cfg['device']
    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    #  note: if you're doing multi target classification, use nn.BCEWithLogitsLoss() and convert labels to float
    # criterion = nn.CrossEntropyLoss()
    """
Below is a code snippet from Tarun, which shows how to use weights in the loss function:
    """
    criterion = nn.CrossEntropyLoss()
    # num_examples = [200000, 500]
    # weights = torch.tensor([max(num_examples)/200000, max(num_examples)/500])
    # criterion = nn.CrossEntropyLoss(weights)


    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)
        # if idx == 0:
        #     print(f"{data.shape=} {labels=}")

        #     im = Image.fromarray((data[0].detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8))
        #     im.save("train.jpeg")
        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    """
Below is a snippet from Tarun, which shows how to use weights in the loss function:
    """
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss
    # num_examples = [200000, 500]
    # weights = torch.tensor([max(num_examples)/200000, max(num_examples)/500])
    # criterion = nn.CrossEntropyLoss(weights)

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)
            # forward pass
            prediction = model(data)

            # if idx == 0:
            #     print(f"{data.shape=} {labels=}")

            #     im = Image.fromarray((data[0].detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8))
            #     im.save("val.jpeg")

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'


    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
        project="cv4e_test",

    # track hyperparameters and run metadata
        config=cfg
        
    )
    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')

    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    if not cfg['only_validate']:
        numEpochs = cfg['num_epochs']
    else:
        numEpochs = 1

    best_val_loss = 1e6
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        if not cfg['only_validate']:
            loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val = validate(cfg, dl_val, model) #todo put back MAH

        # combine stats and save
        # log metrics to wandb
        if not cfg['only_validate']:
            log_dict = {'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
            'lr': optim.param_groups[0]['lr']
        }    
        else:
            log_dict = {'loss_val': loss_val,
            'oa_val': oa_val}

        wandb.log(log_dict)
        stats = log_dict
        best_val_loss = save_model(cfg, current_epoch, model, stats, loss_val, best_val_loss)
    
    #Now we rename the folder model_states to a timestamp (this code by Peter):
    os.rename('model_states', 'model_states_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    wandb.finish()

    # That's all, folks!
        


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
