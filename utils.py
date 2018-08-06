from io import open
import glob # 文件路径查找
import os
import unicodedata
import string
import time
import math
import random
import torch

all_letters=string.ascii_letters+".,;'-"
# abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;'-


# 将Unicode转为ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
# print(unicodeToAscii('Málaga')) # Malaga


# 从文件读取所有行
def read_lines(filename):
    # lines=open(filename,'r',encoding='utf-8').read().strip().split('\n')
    lines=open(filename,'r',encoding='utf-8').readlines()
    return [unicode_to_ascii(line) for line in lines]
# print(readLines('data/names/Arabic.txt'))


# 构建 category_lines dictionary，每一类名字对一个列表
def find_files(path):
    return glob.glob(path)


def load_data():
    n_letters = len(all_letters) + 1  # 加1：EOS
    category_lines={}
    all_categories=[]

    for filename in find_files('data/names/*.txt'):
        category=os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines=read_lines(filename)
        category_lines[category]=lines
    n_categories=len(all_categories)
    if n_categories==0:
        raise RuntimeError("没有找到数据")

    # print("# categories:",n_categories,all_categories)
    return n_letters,n_categories,all_letters,all_categories,category_lines


n_letters,n_categories,all_letters,all_categories,category_lines=load_data()


# 随机选取
def random_choice(l):
    return l[random.randint(0,len(l)-1)]


# 随机选取一类，并获取该类对应的随机名字
def random_training_pair():
    category=random_choice(all_categories)
    line=random_choice(category_lines[category])
    return category,line

# 将category编码成one-hot向量
def category_to_tensor(category):
    li=all_categories.index(category)
    tensor=torch.zeros(1,n_categories)
    tensor[0][li]=1
    return tensor

# 给输入的序列one hot编码
def input_tensor(line):
    tensor=torch.zeros(len(line),1,n_letters)
    for li in range(len(line)):
        letter=line[li]
        tensor[li][0][all_letters.find(letter)]=1
    return tensor

# LongTensor of second letter to end (EOS) for target
def target_tensor(line):
    letter_indexes=[all_letters.find(line[li]) for li in range(1,len(line))]
    letter_indexes.append(n_letters-1) # EOS
    return torch.LongTensor(letter_indexes)

# 创建随机的category, input和 target tensors
def random_train_example():
    category, line = random_training_pair()
    category_tensor = category_to_tensor(category)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)