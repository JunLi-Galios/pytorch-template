import torch
import torch.nn as nn

def L1():
    return nn.L1Loss()

def MSE():
    return nn.MSELoss()

def CrossEntropy():
    return nn.CrossEntropyLoss()

def NLL():
    return nn.NLLLoss()

