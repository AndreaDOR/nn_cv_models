"""
生成训练集和测试集，保存在txt文件中
"""

import os
import random
import sys

train_ratio = 0.6  # 60%作为训练集 40%作为测试集
test_ratio = 1 - train_ratio

rootdata = r"data"
train_list, test_list = [], []
data_list = []

class_flag = -1
classes = ["其他", "露娜"]

for a, b, c in os.walk(rootdata):
    # os.walk会遍历rootdata目录及其所有子目录，并返回一个生成器，
    # 该生成器生成的是3元组，每个3元组包含当前目录路径（a）、当前目录下的所有目录名（b）
    # 和当前目录下的所有文件名（c）。

    for i in range(len(c)):
        data_list.append(os.path.join(a, c[i]))

    # os.path.join负责将多个合并成一个路径，并且里面的正反斜线不会影响路径
    for i in range(0, int(len(c) * train_ratio)):
        # a,c代表这训练集的名字， class_flag代表了训练的标签
        train_data = os.path.join(a, c[i]) + "\t" + str(class_flag) + "\n"
        train_list.append(train_data)

    for i in range(int(len(c) * train_ratio), len(c)):
        test_data = os.path.join(a, c[i]) + "\t" + str(class_flag) + "\n"
        test_list.append(test_data)

    class_flag += 1

random.shuffle(train_list)
random.shuffle(test_list)

# 写入数据到txt中
with open("train.txt", "w", encoding="UTF-8") as f:
    for train_img in train_list:
        f.write(str(train_img))

with open("test.txt", "w", encoding="UTF-8") as f:
    for test_img in test_list:
        f.write(test_img)
