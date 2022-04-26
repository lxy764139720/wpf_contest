import torch


def wpf_loss(output, target):
    loss1 = torch.nn.MSELoss()(output, target)
    loss2 = torch.nn.L1Loss()(output, target)
    return (torch.sqrt(loss1) + loss2) / 2
