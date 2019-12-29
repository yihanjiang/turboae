__author__ = 'yihanjiang'

import torch
import torch.nn.functional as F

class Interleaver(torch.nn.Module):
    def __init__(self, args, p_array):
        super(Interleaver, self).__init__()
        self.args = args
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

    def set_parray(self, p_array):
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

    def forward(self, inputs):

        inputs = inputs.permute(1,0,2)
        res    = inputs[self.p_array]
        res    = res.permute(1, 0, 2)

        return res


class DeInterleaver(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DeInterleaver, self).__init__()
        self.args = args

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(len(p_array))

    def set_parray(self, p_array):

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(len(p_array))

    def forward(self, inputs):
        inputs = inputs.permute(1,0,2)
        res    = inputs[self.reverse_p_array]
        res    = res.permute(1, 0, 2)

        return res

# TBD: change 2D interleavers
# 2D interleavers seems not working well... Don't know why...
class Interleaver2Dold(torch.nn.Module):
    def __init__(self, args, p_array):
        super(Interleaver2D, self).__init__()
        self.args = args
        self.p_array = torch.LongTensor(p_array).view(len(p_array))#.view(args.img_size, args.img_size)

    def set_parray(self, p_array):
        self.p_array = torch.LongTensor(p_array).view(len(p_array))#.view(self.args.img_size, args.img_size)

    def forward(self, inputs):
        input_shape = inputs.shape

        inputs = inputs.view(input_shape[0], input_shape[1], input_shape[2]*input_shape[3])
        inputs = inputs.permute(2, 0, 1)
        res    = inputs[self.p_array]


        res    = res.permute(1, 2, 0)
        res    = res.view(input_shape)

        return res

class DeInterleaver2Dold(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DeInterleaver2D, self).__init__()
        self.args = args

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(self.args.img_size**2)

    def set_parray(self, p_array):

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(self.args.img_size**2)

    def forward(self, inputs):
        input_shape = inputs.shape

        inputs = inputs.view(input_shape[0], input_shape[1], input_shape[2]* input_shape[3])


        inputs = inputs.permute(2,0,1)
        res    = inputs[self.reverse_p_array]

        res    = res.permute(1,2,0)
        res    = res.view(input_shape)

        return res


# TBD: change 2D interleavers
# Play with real 2D interleavers: p_array with 2-step interleaving.
class Interleaver2D(torch.nn.Module):
    def __init__(self, args, p_array):
        super(Interleaver2D, self).__init__()
        self.args = args
        self.p_array = torch.LongTensor(p_array).view(len(p_array))#.view(args.img_size, args.img_size)

    def set_parray(self, p_array):
        self.p_array = torch.LongTensor(p_array).view(len(p_array))#.view(self.args.img_size, args.img_size)

    def forward(self, inputs):
        input_shape = inputs.shape

        inputs = inputs.view(input_shape[0], input_shape[1], input_shape[2]*input_shape[3])
        inputs = inputs.permute(2, 0, 1)
        res    = inputs[self.p_array]


        res    = res.permute(1, 2, 0)
        res    = res.view(input_shape)

        return res

class DeInterleaver2D(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DeInterleaver2D, self).__init__()
        self.args = args

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(self.args.img_size**2)

    def set_parray(self, p_array):

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(self.args.img_size**2)

    def forward(self, inputs):
        input_shape = inputs.shape

        inputs = inputs.view(input_shape[0], input_shape[1], input_shape[2]* input_shape[3])


        inputs = inputs.permute(2,0,1)
        res    = inputs[self.reverse_p_array]

        res    = res.permute(1,2,0)
        res    = res.view(input_shape)

        return res
