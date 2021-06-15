import torch
import numpy as np

def meanEuclideanLoss(pred, gt, scale, jointNr=21):
    pred = pred.view([-1, jointNr, 3])
    gt = gt.reshape([-1, jointNr, 3])
    eucDistance = torch.squeeze(torch.sqrt(torch.sum((pred - gt) ** 2, dim=2)))
    meanEucDistance_normed = torch.mean(eucDistance)
    #print('scale',scale.shape,scale)
    eucDistance = eucDistance * torch.squeeze(scale).view(scale.shape[0], 1)
    meanEucDistance = torch.mean(eucDistance)
    return meanEucDistance_normed, meanEucDistance


def pose_loss(p0, p1, scale,jointN=21):
    pose_loss_rgb = torch.sum((p0 - p1) ** 2, dim=2)
    _, eucLoss_rgb = meanEuclideanLoss(p0, p1, scale,jointN)
    return torch.mean(pose_loss_rgb), eucLoss_rgb


def getLatentLoss(z_mean, z_stddev, goalStd=1.0, eps=1e-9):
    latent_loss = 0.5 * torch.sum(z_mean**2 + z_stddev**2 - torch.log(z_stddev**2)  - goalStd, 1)
    return latent_loss


class LossHelper():
    def __init__(self,precision=3):
        self.loss={}
        self.precision=int(precision)
    def add(self,dic):
        for name,value in dic.items():
            if(name in self.loss):
                self.loss[name].append(float(value))
            else:
                self.loss[name]=[float(value)]
    def show(self):
        for name in self.loss:
            print(name,':loss',np.mean(self.loss[name]))
    def showcurrent(self):
        for name in self.loss:
            txt='{0:.'+str(self.precision)+'f}'
            print(name,txt.format(self.loss[name][-1]),end=" ")
        print()
    def clear(self):
        self.loss={}