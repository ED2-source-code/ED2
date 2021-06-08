import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import math
import copy

table_score = None
all_dimension = None



def omega(G_i, G_j):
    return len(G_i) * len(G_j)

def Dis(G_i, G_j):
    global table_score
    result_list = []
    for i in G_i:
        for j in G_j:
            result_list.append(table_score[int(i)][int(j)])
    return np.mean(np.array(result_list))

def Rela(G_i, G_j):

    global all_dimension

    G_i_ = list(set(all_dimension.copy()) - set(G_i))
    G_j_ = list(set(all_dimension.copy()) - set(G_j))

    bad_score = omega(G_i, G_j_) * Dis(G_i, G_j_) + omega(G_j, G_i_) * Dis(G_j, G_i_)
    bad_num = omega(G_i, G_j_) + omega(G_j, G_i_)

    good_score = omega(G_i, G_j) * Dis(G_i, G_j)
    good_num = omega(G_i, G_j)

    return bad_score / bad_num - good_score / good_num

def get_max_Rela(G):
    max_value = -10000000
    max_groups = None
    for G_i in G:
        for G_j in G:
            if G_i == G_j:
                continue
            if Rela(G_i, G_j) > max_value:
                max_value = Rela(G_i, G_j)
                max_groups = [G_i, G_j]
    return max_value, max_groups

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T) / (a_norm * b_norm)
    dist = -similiarity
    return dist


def get_score(path):
    data = pd.read_excel(path).values[:, 1:]
    data_abs = np.maximum(data, -data)
    x, y = data_abs.shape
    delete_list = []
    for i in range(y):
        if math.isnan(data_abs[0][i]):
            delete_list.append(i)
    data_abs = np.delete(data_abs, delete_list, -1)
    return data_abs



def main(args):
    data = get_score(args.path)
    global table_score
    action_num = data.shape[0]
    table_score_list = []
    for i in range(action_num):
        table_score_list.append([])
        for j in range(action_num):
            table_score_list[i].append(cosine_distance(data[i], data[j]))
    table_score = np.array(table_score_list)
    print("table_score:{}".format(table_score))
    global all_dimension
    all_dimension = [_ for _ in range(action_num)]

    G = [[_] for _ in range(data.shape[0])]
    print(G)
    while len(G) > 1:
        value, group = get_max_Rela(G)
        print(group)
        new_group = group[0].copy()
        new_group.extend(group[1].copy())
        print(new_group)
        G.append(new_group)
        G.remove(group[0])
        G.remove(group[1])
        print(G, value)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help='excel_path')
    args = parser.parse_args()
    print('use parameters: {}'.format(args))
    main(args)
