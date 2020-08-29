import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class DL(nn.Module):
    def __init__(self, smooth=1,square=False):
        super(DL, self).__init__()
        self.smooth = smooth
        self.square = square

    def forward(self, predictions, targets):   
        inputs = predictions[:,1]
        input_flat = inputs.view(-1)
        target_flat = targets.view(-1)
        mult = (input_flat * target_flat).sum()
        if self.square:
            input_flat = torch.mul(input_flat,input_flat)
            target_flat = torch.mul(target_flat,target_flat)
        dice = (2.*mult + self.smooth)/(input_flat.sum() + target_flat.sum() + self.smooth)
        return 1-dice
    

class DSC(nn.Module):
    def __init__(self, smooth=1):
        super(DSC, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        inputs = predictions[:,1]
        input_flat = inputs.view(-1)
        target_flat = targets.view(-1)
        mult = (input_flat * target_flat).sum()
        dice_upper = ((1-input_flat) * mult).sum() + self.smooth
        dice_bottom = ((1-input_flat) * input_flat).sum() + target_flat + self.smooth
        dice = dice_upper/dice_bottom
        return 1-dice
