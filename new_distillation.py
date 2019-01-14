import gensim
#import sklearn
import math
import sys
import os
import pickle
from scipy import spatial

if len(sys.argv) > 1:
    model_num = sys.argv[1]
else:
    print ("Using : python new_distillation.py [model_number]")
    sys.exit()

def cos_sim(list_1, list_2):
    return 1 - spatial.distance.cosine(list_1, list_2)

def dist_maker(dictionary):
    total_dist = []
    #print (dictionary[0])
    for index in range(10):
        if index in dictionary.keys() and dictionary[index] > 0:
            total_dist.append(dictionary[index])
        else:
            total_dist.append(0)
    return total_dist

# Generate output list
output_file = open("distill_files/input_{}.txt".format(str(model_num)), 'r')

output_list = []
all_list = []

for line in output_file:
    all_list.append(line)
    if line not in output_list:
        output_list.append(line)
output_file.close()
print ("output list length :", len(output_list))

# Find index in dictionary
index_dictionary = {}

index = 0
for line in all_list:
    if line not in index_dictionary:
        index_dictionary[line] = [index]
    else:
        index_dictionary[line].append(index)
    index += 1
index_dictionary = sorted(index_dictionary.items(), key=lambda x: len(x[1]),reverse=True)
print ("index list length :", len(index_dictionary))

# Calculating input similarity
similarity_list = []
input_file = open("distill_files/input_{}_emb.pkl".format(model_num), 'rb')
input_embed = pickle.load(input_file)
input_file.close()

sim_file = open("distill_files/similarity_list_{}".format(model_num), 'w')
sim_dist = open("distill_files/similar_distribution_{}".format(model_num), 'w')

i = 0
threshold_num = 20
total = []

for line in index_dictionary:
    length = len(line[1])
    temp_dist = {}
    if length<=threshold_num:break
    ave_sim = 0
    count = 0
    for r_index in range(len(line[1])):
        if r_index != len(line[1]):
            for c_index in range(r_index, len(line[1])):
                sim = cos_sim(input_embed[line[1][r_index]], input_embed[line[1][c_index]])
                temp_sim = max(0.0, math.floor(sim * 10))
                temp_sim = min(9.9999, temp_sim)
                temp_sim = math.floor(temp_sim)
                if temp_sim not in temp_dist:
                    temp_dist[temp_sim] = 1
                else:
                    temp_dist[temp_sim] += 1
                ave_sim += sim
                count += 1
    ave_sim = float((ave_sim-length)/(count-length))
    sorting = dist_maker(temp_dist)
    print ("ave_sim : ", ave_sim)
    print ("count : ", count-length)
    print ("length : ", length)
    similarity_list.append(ave_sim)
    total.append(sorting)
    sim_file.write(str(ave_sim))
    sim_dist.write(str(sorting))
    sim_file.write("\t")
    sim_dist.write("\t")
    sim_file.write(str(line[1]))
    sim_file.write("\t")
    sim_file.write(str(line[0]).strip())
    sim_dist.write(str(line[0]).strip())
    sim_file.write("\n")
    sim_dist.write("\n")

sim_file.close()
sim_dist.close()

sim_pickle = open("distill_files/similar_pickle_{}".format(model_num), 'wb')
pickle.dump(total, sim_pickle)
sim_pickle.close()
