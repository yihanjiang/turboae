__author__ = 'yihanjiang'
import torch
import torch.nn.functional as F

# utility for Same Shape CNN 1D
class SameShapeConv1d(torch.nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, kernel_size, activation = 'elu', no_act = False):
        super(SameShapeConv1d, self).__init__()

        self.cnns = torch.nn.ModuleList()
        self.num_layer = num_layer
        self.no_act = no_act
        for idx in range(num_layer):
            if idx == 0:
                self.cnns.append(torch.nn.Conv1d(in_channels = in_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=1, padding=(kernel_size // 2),
                                                      dilation=1, groups=1, bias=True)
                )
            else:
                self.cnns.append(torch.nn.Conv1d(in_channels = out_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=1, padding=(kernel_size // 2),
                                                      dilation=1, groups=1, bias=True)
                )

        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'selu':
            self.activation = F.selu
        elif activation == 'prelu':
            self.activation = F.prelu
        else:
            self.activation = F.elu

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1,2)
        x = inputs
        for idx in range(self.num_layer):
            if self.no_act:
                x = self.cnns[idx](x)
            else:
                x = self.activation(self.cnns[idx](x))

        outputs = torch.transpose(x, 1,2)
        return outputs


class DenseSameShapeConv1d(torch.nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, kernel_size):
        super(DenseSameShapeConv1d, self).__init__()

        self.cnns = torch.nn.ModuleList()
        self.num_layer = num_layer
        for idx in range(num_layer):
            if idx == 0:
                self.cnns.append(torch.nn.Conv1d(in_channels = in_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=1, padding=(kernel_size // 2),
                                                      dilation=1, groups=1, bias=True)
                )
            else:
                self.cnns.append(torch.nn.Conv1d(in_channels = in_channels + idx * out_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=1, padding=(kernel_size // 2),
                                                      dilation=1, groups=1, bias=True)
                )

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1,2)

        for idx in range(self.num_layer):
            if idx == 0:
                this_input = inputs
            else:
                this_input = torch.cat([this_input, output], dim=1)


            x = self.cnns[idx](this_input)
            output = F.elu(x)


        outputs = torch.transpose(output, 1,2)
        return outputs


###################################################
# utility for Same Shape CNN 2D: experimental!
# not working very well yet!
###################################################
class SameShapeConv2d(torch.nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, kernel_size,  no_act = False):
        super(SameShapeConv2d, self).__init__()
        self.no_act = no_act
        self.cnns = torch.nn.ModuleList()
        self.num_layer = num_layer
        for idx in range(num_layer):
            if idx == 0:
                self.cnns.append(torch.nn.Conv2d(in_channels = in_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=1, padding=(kernel_size // 2),
                                                      dilation=1, groups=1, bias=True)
                )
            else:
                self.cnns.append(torch.nn.Conv2d(in_channels = out_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=1, padding=(kernel_size // 2),
                                                      dilation=1, groups=1, bias=True)
                )


    def forward(self, inputs):

        x = inputs
        for idx in range(self.num_layer):
            if self.no_act:
                x = self.cnns[idx](x)
            else:
                x = F.elu(self.cnns[idx](x))

        return x




class DenseSameShapeConv2d(torch.nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, kernel_size,  no_act = False):
        super(DenseSameShapeConv2d, self).__init__()
        self.no_act = no_act
        self.cnns = torch.nn.ModuleList()
        self.num_layer = num_layer
        for idx in range(num_layer):
            if idx == 0:
                self.cnns.append(torch.nn.Conv2d(in_channels = in_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=1, padding=(kernel_size // 2),
                                                      dilation=1, groups=1, bias=True)
                )
            else:
                self.cnns.append(torch.nn.Conv2d(in_channels = in_channels + idx * out_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=1, padding=(kernel_size // 2),
                                                      dilation=1, groups=1, bias=True)
                )

    def forward(self, inputs):

        x = inputs
        for idx in range(self.num_layer):
            if idx == 0:
                this_input = inputs
            else:
                this_input = torch.cat([this_input, output], dim=1)

            if self.no_act:
                output = self.cnns[idx](this_input)
            else:
                output = F.elu(self.cnns[idx](this_input))

        return output

