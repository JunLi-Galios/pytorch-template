import torch

def SGD(parameters, learning_rate, momentum, weight_decay):
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, nesterov=True, weight_decay=weight_decay)

def ADAM(parameters, learning_rate, momentum, weight_decay):
    return torch.optim.Adam(parameters, lr=learning_rate, betas=(momentum, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
