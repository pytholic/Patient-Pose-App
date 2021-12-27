import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
  def __init__(self, imgs_path=str, transforms=None):
    self.imgs_path = imgs_path
    self.transforms = transforms
    file_list = glob.glob(self.imgs_path + "*")
    self.data = []
    for class_path in file_list:
      # for windows
      tmp = class_path.split("/")[-1]
      class_name = tmp.split("\\")[-1]
      for img_path in glob.glob(class_path + "/*.png"):
        self.data.append([img_path, class_name])
    self.class_map = {"head_left" : 0, "head_right": 1, "none": 2, "standing": 3}

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_path, class_name = self.data[idx]
    img = cv2.imread(img_path, 1)
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    img = img.float()
    label = self.class_map[class_name]

    #Applying transforms on image
    if self.transforms:
      img = self.transforms(img)

    return img, label

if __name__ == "__main__":
  dataset = CustomDataset(imgs_path='./dataset/train/')   
  data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
  for imgs, labels in data_loader:
    print("Batch of images has shape: ",imgs.shape)
    print("Batch of labels has shape: ", labels.shape)

