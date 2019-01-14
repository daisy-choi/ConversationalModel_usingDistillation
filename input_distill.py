import pickle
import os
import sys

if len(sys.argv) > 1:
    model_num = sys.argv[1] # convert to num -> num+1
else:
    print ("Using : python input_distill.py [model_number]")
    sys.exit()

threshold = 0.7

# Open original file
original_file = open("distill_files/src-train.{}".format(model_num), "r")
target_file = open("distill_files/tar-train.{}".format(model_num), "r")

original_input = original_file.readlines()
target_input = target_file.readlines()

original_file.close()
target_file.close()

print ("original file length :", len(original_input))

sim_file = open("distill_files/similarity_list_{}".format(model_num), "r")
remove_file = open("distill_files/Response{}.txt".format(model_num), 'w')

remove_list = []

for line in sim_file:
    tokens = line.split("\t")
    if float(tokens[0]) < threshold:
        temp_line = tokens[1].replace("[","").replace("]","").split(", ")
        remove_file.write(tokens[2])
        if len(temp_line) < 50:
            break
        for index in temp_line:
            remove_list.append(float(index))

remove_file.close()
sim_file.close()
print ("The number of removed files :", len(remove_list))

# Remove files
if int(model_num) < 3:
    new_file = open("distill_files/src-train.{}".format(int(model_num) + 1), "w")
    new_tar = open("distill_files/tar-train.{}".format(int(model_num) + 1), "w")

    for index in range(len(original_input)):
        if index not in remove_list:
            new_file.write(original_input[index])
            new_tar.write(target_input[index])

    new_file.close()
    new_tar.close()
