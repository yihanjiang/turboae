__author__ = 'yihanjiang'
import pyldpc
import numpy as np
from utilities import hamming_dist

def ldpc_regularldpc_encoder(message, tG, snr):

    y = pyldpc.Coding(tG,message,snr)
    return y

def ldpc_regularldpc_decoder(received, H,tG,  snr, max_iter):
    x_decoded = pyldpc.Decoding_logBP(H,received,snr,max_iter)
    v_received = pyldpc.DecodedMessage(tG,x_decoded)
    return v_received


## define LDPC parameters

n = 15  # Number of columns
d_v = 3 # Number of ones per column, must be lower than d_c (because H must have more rows than columns)
d_c = 5 # Number of ones per row, must divide n (because if H has m rows: m*d_c = n*d_v (compute number of ones in H))

max_iter = 10

num_block = 100

H = pyldpc.RegularH(n,d_v,d_c)
tG = pyldpc.CodingMatrix(H)

n,k = tG.shape

print 'LDPC has shape', n, k

snrs = [0.5*item for item in range(-3, 5)]

print snrs

train_X, train_Y = [], []
train_snr = 0.0
for _ in range(num_block):
    message  = np.random.randint(2,size=k)
    received = ldpc_regularldpc_encoder(message, tG, train_snr)
    train_Y.append(received)
    train_X.append(message)

train_X = np.array(train_X)
train_Y = np.array(train_Y)

print train_X.shape, train_Y.shape


# for snr in snrs:
#     this_bit_err = 0
#     this_block_err = 0
#
#     for _ in range(num_block):
#         message  = np.random.randint(2,size=k)
#         received = ldpc_regularldpc_encoder(message, tG, snr)
#         decoded  = ldpc_regularldpc_decoder(received, H,tG, snr, max_iter)
#
#         num_err  = hamming_dist(message,decoded )
#         this_bit_err += num_err
#         if num_err !=0:
#             this_block_err += 1
#
#     this_ber = this_bit_err*1.0/(num_block*k)
#     this_bler = this_block_err*1.0/num_block
#
#     print this_ber, this_bler
