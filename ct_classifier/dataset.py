'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
from torch.utils.data import Dataset
# from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms import v2
from PIL import Image


class CTDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        # Transforms. Here's where we could add data augmentation 
        #  For now, we just resize the images to the same dimensions...and convert them to torch.Tensor.
        #  For other transformations see Björn's lecture on August 11 or 
        # https://pytorch.org/vision/stable/transforms.html <- We looked at this with Val on 17.01.25 and adjusted to the v2 
        # version of the transforms module.
        if split == 'train':
            self.transform = v2.Compose([              
                v2.Resize((cfg['image_size'])), 
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),       
                v2.ToTensor()                          
            ])
        else:
            self.transform = v2.Compose([
                v2.Resize((cfg['image_size'])),
                v2.ToTensor()])
        
        # index data into list
        self.data = []

        # load annotation file - Modified 14.01.24 by Talen / Julia
        if split == 'train':
            annot_file = 'train_annotations.json'
        elif split == 'val':
            annot_file = 'cis_val_annotations.json'
        elif split == 'test':
            annot_file = 'cis_test_annotations.json'
        else:
            raise ValueError(f'Invalid split "{split}"')
        annoPath = os.path.join(
            self.data_root,
            'eccv_18_annotation_files',
            annot_file
        )

        # # load annotation file

        # annoPath = os.path.join(
        #     self.data_root,
        #     'eccv_18_annotation_files',
        #     'train_annotations.json' if self.split=='train' else 'cis_val_annotations.json'
        # )
        meta = json.load(open(annoPath, 'r'))

        # enable filename lookup. Creates image IDs and assigns each ID one filename. 
        #  If your original images have multiple detections per image, this code assumes
        #  that you've saved each detection as one image that is cropped to the size of the
        #  detection, e.g., via megadetector.
        images = dict([[i['id'], i['file_name']] for i in meta['images']])
        # create custom indices for each category that start at zero. Note: if you have already
        #  had indices for each category, they might not match the new indices.
        # labels = dict([[c['id'], idx] for idx, c in enumerate(meta['categories'])])
        
        # since we're doing classification, we're just taking the first annotation per image and drop the rest
        images_covered = set()      # all those images for which we have already assigned a label
        for anno in meta['annotations']:
            imgID = anno['image_id']
            if imgID in images_covered:
                continue
            
            # append image-label tuple to data
            imgFileName = images[imgID]
            label = anno['category_id']
            #Use breakpoint() to help debugging (can look around at code and objects before the error occurs)
            # labelIndex = labels[label] #Julia Commented this out

            self.data.append([imgFileName, label])
            images_covered.add(imgID)       # make sure image is only added once to dataset
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = os.path.join(self.data_root, 'eccv_18_all_images_sm', image_name)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)

        return img_tensor, label