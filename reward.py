import pickle

import os


f = open("reward_result.pkl", 'rb')
list = pickle.load(f)
#print (list)
dic = {}
for tensor in list:
    if tensor.item() not in dic:
        dic[tensor.item()] = 1
    else:
        dic[tensor.item()] += 1
print (dic)
f.close()
