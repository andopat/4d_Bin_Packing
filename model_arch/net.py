import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import sys
import config


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class CNNPro(nn.Module):
    def __init__(self, name="train"):

        super(CNNPro, self).__init__()        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        self.name = name
        self.share = nn.Sequential(
            init_(nn.Conv2d(config.channel, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            )

        self.actor = nn.Sequential(
            init_(nn.Conv2d(64, 64, 1, stride=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 16, 1, stride=1)),
            nn.ReLU(),
            init_(nn.Conv2d(16, 2, 1, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(2*config.max_X*config.max_Y, config.actor_hidden)),            
            nn.ReLU(),      
            init_(nn.Linear(config.actor_hidden, config.act_len)),
            nn.ReLU(),             
        )

        self.critic = nn.Sequential(
            init_(nn.Conv2d(64, 64, 1, stride=1)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 16, 1, stride=1)),
            nn.ReLU(),
            init_(nn.Conv2d(16, 2, 1, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(2*config.max_X*config.max_Y, config.critic_hidden)),        
            nn.ReLU(),
            nn.Linear(config.critic_hidden, 1)
        )

        self.train()

    def forward(self, inputs):

        share = self.share(inputs)

        logits = self.actor(share)
        values = self.critic(share)

        return logits, values


