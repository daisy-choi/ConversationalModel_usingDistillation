import pickle
import os
from scipy import spatial

import torch.nn as nn, torch
import torch.nn.init as init

def cos_sim(list1, list2):
    return 1 - spatial.distance.cosine(list1, list2)

def Calculate_score(output, response):
    total_score = []
    for index in range(len(output)):
        max_score = 0
        for res_index in range(len(response)):
            score = cos_sim(output[index], response[res_index])
            if score > max_score:
                max_score = score
        if max_score != 0:
            total_score.append(max_score)
        else:
            total_score.append(0.0001)
    return total_score

def main():
    N = 4
    rele_score_set = []
    for index in range(N):
        # output embedding vectors
        f_out = open("distill_files/output_{}_emb.pkl".format(index), 'rb')
        out_bow = pickle.load(f_out)
        f_out.close()

        # Response embedding vectors
        f_res = open("distill_files/Response_{}_emb.pkl".format(index), 'rb')
        res_bow = pickle.load(f_res)
        f_res.close()

        score_list = Calculate_score(out_bow, res_bow)

        if index == 0:
            for score_index in range(len(score_list)):
                rele_score_set.append([score_list[score_index]])
        else:
            for score_index in range(len(score_list)):
                rele_score_set[score_index].append(score_list[score_index])

    f_rele = open("distill_files/sim_relevance_score.pkl", 'wb')
    pickle.dump(rele_score_set, f_rele)
    f_rele.close()

main()
