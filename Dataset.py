from data_helper import clean_text
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from skimage import io,transform
import torchvision
import torch
#mport torchtext

#https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
class IG_img_caption_dataset(Dataset):
  def __init__(self, csv_file,type_prof,vocab, transform=None):
    self.type = type_prof
    self.annotations = csv_file[csv_file['Type']==self.type]
    self.transform = transform
    self.vocab = vocab
    
  def __len__(self):
    return len(self.annotations)

  def __getitem__(self,index):
    img_path = self.annotations.iloc[index,2]
    vocab = self.vocab
    #image = torch.from_numpy(cv2.resize(cv2.imread(img_path),(256,256)))
    #image = image.astype(np.uint8)
    image = Image.open(img_path)
    #image = io.imread(img_path,plugin='pil')
    #image = transform.resize(image,(256,256))
    caption_path = self.annotations.iloc[index,3]
    caption = clean_text(
      open(caption_path,'r',encoding='utf8').read()
      .split())
    caption = [vocab(token) for token in caption]
    target = torch.Tensor(caption)

    if self.transform:
      image = self.transform(image)

    #sample = {'image':image,'caption':target}
    
    return image,target

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Rescale(object):
  """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
  def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

  def __call__(self,image):
    #image = sample['image']
    h, w = image.shape[:2]
    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    img = transform.resize(image, (new_h, new_w))

    return img

class RandomCrop(object):
  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self,image):
    #image = sample['image']
    h, w = image.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    image = image[top: top + new_h,left: left + new_w]

    return image

class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""

  def __call__(self,image):

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)


