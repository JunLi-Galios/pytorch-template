import torch

def Step(optimizer, step_size, gamma=0.1, last_epoch=-1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


def MultiStep(optimizer, milestones, gamma=0.1, last_epoch=-1):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)

def Constant(optimizer):
    return None