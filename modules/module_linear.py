import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, visual_config, max_frames, max_words):
        super().__init__()
        self.max_frames = max_frames
        input_size = torch.zeros([self.max_frames+max_words, visual_config.hidden_size])
        input_size = input_size.view(-1)
        output_size = torch.zeros([157, self.max_frames])
        output_size = output_size.view(-1)
        self.hidden = nn.Linear(len(input_size), len(output_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = x.reshape([157, self.max_frames])
        return x