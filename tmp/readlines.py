__author__ = 'yihanjiang'
import re
file = open('./724820_log.txt','r')

res = []
for line in file:


    if line[:len('====> Test set BCE loss')] == '====> Test set BCE loss':
        lines = line.strip().split(' ')
        print lines
        item =  float(lines[-1])

        res.append(item)

print res