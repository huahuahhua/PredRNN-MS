import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
import math


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        self.cell_mode = configs.cell_mode
        self.states = ['recall', 'normal']
        if not self.configs.model_mode in self.states:
            raise AssertionError
        self.embed = nn.Conv2d(self.frame_channel, num_hidden[0], kernel_size=1, stride=1, padding=0)
        lstm = [SpatioTemporalLSTMCell(num_hidden[0], num_hidden[0], configs, configs.img_width, configs.img_height),
                SpatioTemporalLSTMCell(num_hidden[0], num_hidden[0], configs, configs.img_width / 2,
                                       configs.img_height / 2),
                SpatioTemporalLSTMCell(num_hidden[0], num_hidden[0], configs, configs.img_width / 4,
                                       configs.img_height / 4),
                SpatioTemporalLSTMCell(num_hidden[0], num_hidden[0], configs, configs.img_width / 4,
                                       configs.img_height / 4),
                SpatioTemporalLSTMCell(num_hidden[0], num_hidden[0], configs, configs.img_width / 2,
                                       configs.img_height / 2),
                SpatioTemporalLSTMCell(num_hidden[0], num_hidden[0], configs, configs.img_width, configs.img_height)]
        self.cell_list = nn.ModuleList(lstm)

        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups = nn.ModuleList(
            [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        self.downs_m = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups_m = nn.ModuleList(
            [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        self.fc = nn.Conv2d(self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

        print('This is Multi Scale PredRNN!')

    def forward(self, frames, mask_true):

        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        batch = frames.shape[0]

        next_frames = []

        zeros = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        zeros2 = torch.zeros([batch, self.num_hidden[0], height / 2, width / 2]).to(self.configs.device)
        zeros4 = torch.zeros([batch, self.num_hidden[0], height / 4, width / 4]).to(self.configs.device)

        h_t = [zeros, zeros2, zeros4, zeros4, zeros2, zeros]
        c_t = [zeros, zeros2, zeros4, zeros4, zeros2, zeros]

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        x_gen = None
        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                time_diff = t - self.configs.input_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
            frames_feature = net

            frames_feature = self.embed(frames_feature)

            for l in range(self.num_layers):

                if l == 0:
                    h_t[0], c_t[0], memory = self.cell_list[0](frames_feature, h_t[0], c_t[0], memory)
                    frames_feature = self.downs[0](h_t[0])
                    memory = self.downs_m[0](memory)
                elif l == 1:
                    h_t[l], c_t[l], memory = self.cell_list[l](frames_feature, h_t[l], c_t[l], memory)
                    frames_feature = self.downs[1](h_t[l])
                    memory = self.downs_m[1](memory)
                elif l == 2:
                    h_t[l], c_t[l], memory = self.cell_list[l](frames_feature, h_t[l], c_t[l], memory)
                    frames_feature = h_t[l]
                elif l == 3:
                    h_t[l], c_t[l], memory = self.cell_list[l](frames_feature, h_t[l], c_t[l], memory)
                    frames_feature = self.ups[0](h_t[l])
                    memory = self.ups_m[0](memory)
                elif l == 4:
                    h_t[l], c_t[l], memory = self.cell_list[l](frames_feature, h_t[l], c_t[l], memory)
                    frames_feature = self.ups[1](h_t[l])
                    memory = self.ups_m[1](memory)
                elif l == 5:
                    h_t[l], c_t[l], memory = self.cell_list[l](frames_feature, h_t[l], c_t[l], memory)

            x_gen = h_t[-1]

            x_gen = self.fc(x_gen)
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames
