# Test.py

# This is the test python file

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np

from dataset import CTDataset
from model import CustomResNet18

def create_dataloader(cfg, split='test'):
    '''Loads dataset and creates dataloader'''
    dataset_instance = CTDataset(cfg, split)
    dataLoader = DataLoader(
        dataset=dataset_instance,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers']
    )
    return dataLoader

def validate(cfg, dataLoader, model):
    '''Validation function copied from train.py'''
    device = cfg['device']
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    loss_total, oa_total = 0.0, 0.0
    correct_class_1_count_total = 0
    class_1_count_total = 0
    class_0_count_total = 0
    correct_class_0_count_total = 0
    
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():
        for idx, (data, labels, image_name) in enumerate(dataLoader):
            print (image_name, labels)

            data, labels = data.to(device), labels.to(device)
            prediction = model(data)
            loss = criterion(prediction, labels)
            
            loss_total += loss.item()
            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()
            
            correct_class_1_count_total += torch.sum(((labels == 1) & (pred_label==1)).float())
            class_1_count_total += torch.sum((labels == 1).float())
            class_0_count_total += torch.sum((labels == 0).float())
            correct_class_0_count_total += torch.sum(((labels == 0) & (pred_label==0)).float())
            
            progressBar.set_description(
                '[Test] Loss: {:.2f}; OA: {:.2f}%; class 1 label count: {:.2f}; class 1 correct pred count: {:.2f}; class 0 label count: {:.2f}; class 0 correct pred count: {:.2f}'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1),
                    class_1_count_total,
                    correct_class_1_count_total,
                    class_0_count_total,
                    correct_class_0_count_total
                )
            )
            progressBar.update(1)
    
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)
    print('############')
    print('class 1 accuracy:', correct_class_1_count_total/class_1_count_total)
    print('class 0 accuracy:', correct_class_0_count_total/class_0_count_total)
    return loss_total, oa_total

def main():
    # Load configuration
    config_path = '../model_states_test/config.yaml'
    cfg = yaml.safe_load(open(config_path, 'r'))
    
    # Check device
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'
    
    # Initialize model and load weights
    model = CustomResNet18(cfg['num_classes'])
    checkpoint = torch.load('../model_states_test/best.pt')
    model.load_state_dict(checkpoint['model'])
    
    # Create test dataloader
    dl_test = create_dataloader(cfg, split='train')
    
    # Run testing
    loss_test, oa_test = validate(cfg, dl_test, model)

if __name__ == '__main__':
    main()