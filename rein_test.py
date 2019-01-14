import argparse
import time
import math
import random
import torch.nn as nn, torch
import torch.nn.init as init
import torch.optim as optim
import os
import numpy as np
import pickle

from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer

class BaseEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size):
        super(BaseEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = 1
        self.drop = nn.Dropout(0.4)
        self.direction = 2
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=0, sparse=False)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size, num_layers=self.num_lyr, bidirectional=True, batch_first=True, dropout=0.4)

    def forward(self, inp):
        x = inp.view(-1, inp.size(2))
        x_emb = self.embed(x)
        x_emb = self.drop(x_emb)

        bt_siz, seq_len = x_emb.size(0), x_emb.size(1)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()
        x_o, x_hid = self.rnn(x_emb, h_0)
        if self.direction == 2:
            x_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(x_hid[2 * i:2 * i + 2, :, :], 0, keepdim=True)
                x_hids.append(x_hid_temp)
            x_hid = torch.cat(x_hids, 0)

        x_hid = x_hid[self.num_lyr - 1, :, :].unsqueeze(0)
        x_hid = x_hid.transpose(0, 1)

        return x_hid


class Policy_Network(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, batch_size):
        super(Policy_Network, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = BaseEncoder(vocab_size, 300, 400).cuda()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout_layer = nn.Dropout(p=0.2)

        self.total_reward = 0
        self.num_reward = 0

        self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        return(torch.randn(1, batch_size, self.hidden_dim).cuda(), torch.randn(1, batch_size, self.hidden_dim).cuda())

    def baseline_score(self, reward, num_reward):
        return reward / num_reward

    def forward(self, input):
        input = self.embedding(input)
        input = input.transpose(0, 1)

        outputs, self.hidden = self.lstm(input, self.hidden)

        output = self.dropout_layer(self.hidden[0][-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        pred_index = (output.max(1)[1])

        return pred_index

def cos_sim(list1, list2):
    return 1 - spatial.distance.cosine(list1, list2)

def stoi(sentence, vocab_list):
    max_token = 248
    result_list = [1]
    tokens = sentence.replace("\n", "").strip().split()
    count = 0
    
    for index in range(max_token - 1):
        if count < len(tokens):
            if tokens[index] in vocab_list:
                result_list.append(vocab_list.index(tokens[index]))
            else:
                result_list = 0
                return result_list
                break
        elif count == len(tokens):
            result_list.append(2)
        else:
            result_list.append(0)
        count += 1
    return result_list


use_cuda = torch.cuda.is_available()
#torch.manual_seed(123)
#np.random.seed(123)
#if use_cuda:
#    torch.cuda.manual_seed(123)

##############################################################################################################################################

def RL_test_model(RL_model, dataloader):
    RL_model.eval()

    total_reward_list = []
    result_output = open("output_result.txt", 'w')
    total_index = 0

    for i_batch, sample_batch in enumerate(dataloader):
        RL_model.hidden = RL_model.init_hidden(len(sample_batch))

        pred_index = RL_model(sample_batch)

        for index in pred_index:
            f = open("temp_files/test_output_{}.txt".format(index), 'r')
            result_output.write(f.readlines()[total_index])
            f.close()
            total_reward_list.append(index)
            total_index += 1

    result_output.close()
    reward_file = open("reward_result.pkl", "wb")
    pickle.dump(total_reward_list, reward_file)
    reward_file.close()

##############################################################################################################################################


def main():
    with open('./distill_files/w2i', 'rb') as f:
        inv_dict = pickle.load(f)

    # parameter
    N = 4
    folder_path = "distill_files/"

    #f = open("demo_temp/src-test.pkl", 'rb')
    #line_list = pickle.load(f)
    #f.close()

    f = open("demo_temp/src_test.txt", 'r')
    #line_list = pickle.load(f)
    line_list = f.readlines()
    f.close()

    f = open("temp", 'w')

    new_list = []
    for line in line_list:
        temp_line = stoi(line, inv_dict)
        if temp_line != 0:
            f.write(line)
            new_list.append(Variable(torch.LongTensor([temp_line])).cuda())

    f.close()

    #new_list = []
    #for line in line_list:
    #    new_list.append(Variable(torch.LongTensor([line])).cuda())
    #print (len(new_list))

    dataloader = DataLoader(new_list, 64, shuffle=False)

    RL_model = Policy_Network(len(inv_dict), 400, 128, N, 64).cuda()
    optimizer = optim.SGD(RL_model.parameters(), lr=0.1, weight_decay=1e-4)

    RL_model.load_state_dict(torch.load("save/new_RL_model_epoch199.pt"))

    RL_test_model(RL_model, dataloader)

main()
