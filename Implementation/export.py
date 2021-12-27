import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler

import torchvision.transforms as T
import torchvision.models as models
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

from matplotlib import pyplot as plt
from dataloader import *
from __main__ import *

<<<<<<< HEAD
import onnx
#import coremltools as ct
=======
>>>>>>> bb84d6fba44411619b87889293b4569d7d46f7df

if model_dir:
  MODEL_DIR = model_dir[0]
else:
  MODEL_DIR = None

name = 'models'
modelDir = os.path.join(os.getcwd(), f'{name}')

# # for resnet
# model = models.resnet18(pretrained=True)
# num_features = model.fc.in_features
# num_classes = 4
# model.fc = nn.Linear(num_features, num_classes)
# print(model)

# for mobilenet
model = models.mobilenet_v2(pretrained=True)
num_features = model.classifier[1].in_features
num_classes = 4
model.classifier[1] = nn.Linear(num_features, num_classes)
print(model)

# Load the trained model
model = model
state_dict = None

if MODEL_DIR:
  state_dict = torch.load(os.path.join(MODEL_DIR, 'best_model.pth'), map_location=torch.device('cpu'))
<<<<<<< HEAD
elif os.path.exists(modelDir):
=======
if os.path.exists(modelDir):
>>>>>>> bb84d6fba44411619b87889293b4569d7d46f7df
  state_dict = torch.load(os.path.join(modelDir, 'best_model.pth'), map_location=torch.device('cpu')) # for CPU
else:
  print("Model file does not exist. Please enter path for model file.")

try:
  model.load_state_dict(state_dict)
except:
    print("No model found, state_dict is empty!")
    sys.exit(1)


# We don't need gpu for inference
device = torch.device("cpu")
model.to(device)

<<<<<<< HEAD
if os.path.exists(modelDir):
  path = modelDir
elif MODEL_DIR:
  path = MODEL_DIR
else:
  print("Save directory not found!")
=======
>>>>>>> bb84d6fba44411619b87889293b4569d7d46f7df

# Onnx export
input = torch.randn(1, 3, 240, 320)
torch.onnx.export(model, 
                  input, 
<<<<<<< HEAD
                  os.path.join(path, 'model.onnx'), 
=======
                  os.path.join(modelDir, 'model.onnx'), 
>>>>>>> bb84d6fba44411619b87889293b4569d7d46f7df
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names = ['X'], 
                  output_names = ['Y']
                  )


<<<<<<< HEAD


# CoreML Export
# model_onnx = onnx.loados.path.join(path, 'model.onnx')
# model_coreml  = ct.converters.onnx.convert(model=model_onnx)

# # Save model
# model_coreml.save(os.path.join(path, "model.mlmodel"))

# input = torch.randn(1, 3, 240, 320)
# model = model.eval()
# traced_model = torch.jit.trace(model, input)

# import coremltools as ct

# # Convert to Core ML using the Unified Conversion API
# scale = 1/(0.5*255.0)
# bias = [- 0.5/(0.5) , - 0.5/(0.5), - 0.5/(0.5)]

# model = ct.convert(
#     traced_model,
#     inputs=[ct.ImageType(name="input_1",
#                         shape=input.shape,
#                         scale=scale,
#                         bias=bias)]) 

# # inputs=[ct.TensorType(name="input_1", shape=input.shape)]

# # Save model
# # if os.path.exists(modelDir):
# #     model.save(os.path.join(modelDir, "model.mlmodel"))
# # elif MODEL_DIR:
# #     model.save(os.path.join(MODEL_DIR, "model.mlmodel"))
# # else:
# #     print("Save directory not found!")
=======
# CoreML Export
input = torch.randn(1, 3, 240, 320)
model = model.eval()
traced_model = torch.jit.trace(model, input)

import coremltools as ct

# Convert to Core ML using the Unified Conversion API
scale = 1/(0.5*255.0)
bias = [- 0.5/(0.5) , - 0.5/(0.5), - 0.5/(0.5)]

model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input_1",
                        shape=input.shape,
                        scale=scale,
                        bias=bias)]) 

# inputs=[ct.TensorType(name="input_1", shape=input.shape)]

# Save model
if os.path.exists(modelDir):
    model.save(os.path.join(modelDir, "model.mlmodel"))
elif MODEL_DIR:
    model.save(os.path.join(MODEL_DIR, "model.mlmodel"))
else:
    print("Save directory not found!")
>>>>>>> bb84d6fba44411619b87889293b4569d7d46f7df
