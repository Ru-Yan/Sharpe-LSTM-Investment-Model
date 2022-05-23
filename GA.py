#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: liujie
@software: PyCharm
@file: GA.py
@time: 2020/11/17 22:28
"""
'''
    在优化神经网络上，用常规的遗传算法不易实现
    原因如下：
        1.传统的遗传算法中每条染色体的长度相同，但是优化LSTM网络时染色体的长度会因为层数的不同而不同
          比如：a染色体有一层LSTM层和一层全连接层，则这个染色体上共有6个基因(两个代表层数，两个代表神经元个数)
               b染色体有二层LSTM层和二层全连接层，则这个染色体上共有6个基因(两个代表层数，四个代表每层的神经元个数)
               
        2.在传统的遗传算法中，染色体上的基因的取值范围是相同的，但是在优化LSTM网络时，需要让表示层数的基因在一个范围内，
          表示神经元个数的基因在另一个范围内，比如层数范围是一到三层，神经元个数是32到256个之间
        3.由于染色体长度不同，交叉函数、变异函数均需要做出修改
        
    解决办法：
        1.将每条染色体设置为相同的长度
          (本题来说，因为LSTM层与全连接层层数最多三层，加上最前面两个表示层数的基因，故每条染色体上有3+3+2 = 8个基因)，
          达不到长度要求的后面补零
        2.先设置前面两个基因，令其范围分别在一到三之间，然后根据这两个基因确定后面关于每层神经元个数的基因
        3.对于交叉函数的修改，首先确定取出的两条染色体(设为a染色体和b染色体)上需要交换的位置，然后遍历两条染色体在这些位置的
          基因，如果任一染色体上此位置上的基因为0或要交换的基因是关于层数的，则取消此位置的交换
        4.对于变异函数的修改，只有关于神经元个数的基因变异，关于层数的基因不变异
'''

import numpy as np
import otherscode as project
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置参数
DNA_size = 2
DNA_size_max = 8    # 每条染色体的长度
POP_size = 20       # 种群数量
CROSS_RATE = 0.5    # 交叉率
MUTATION_RATE = 0.01# 变异率
N_GENERATIONS = 40  # 迭代次数

x_train,y_train,x_test,y_test = project.load()

# 适应度
def get_fitness(x):
    return project.classify(x_train,y_train,x_test,y_test,num=x)

# 生成新的种群
def select(pop,fitness):
    idx = np.random.choice(np.arange(POP_size),size=POP_size,replace=True,p=fitness/fitness.sum())
    return pop[idx]

# 交叉函数
def crossover(parent,pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0,POP_size,size=1)   # 染色体的序号
        cross_points = np.random.randint(0,2,size=DNA_size_max).astype(np.bool)     # 用True、False表示是否置换

        # 对此位置上基因为0或是要交换的基因是关于层数的，则取消置换
        for i,point in enumerate(cross_points):
            if point == True and pop[i_,i] * parent[i] == 0:
                cross_points[i] = False
            # 修改关于层数的
            if point == True and i < 2:
                cross_points[i] = False
        # 将第i_条染色体上对应位置的基因置换到parent染色体上
        parent[cross_points] = pop[i_,cross_points]
    return parent


# 定义变异函数
def mutate(child):
    for point in range(DNA_size_max):
        if np.random.rand() < MUTATION_RATE:
            if point >= 3:
                if child[point] != 0:
                    child[point] = np.random.randint(32,257)
    return child

# 层数
pop_layers = np.zeros((POP_size,DNA_size),np.int32)
pop_layers[:,0] = np.random.randint(1,4,size=(POP_size,))
pop_layers[:,1] = np.random.randint(1,4,size=(POP_size,))

# 种群
pop = np.zeros((POP_size,DNA_size_max))
# 神经元个数
for i in range(POP_size):
    pop_neurons = np.random.randint(32,257,size=(pop_layers[i].sum(),))
    pop_stack = np.hstack((pop_layers[i],pop_neurons))
    for j,gene in enumerate(pop_stack):
        pop[i][j] = gene

# 迭代次数
for each_generation in range(N_GENERATIONS):
    # 适应度
    fitness = np.zeros([POP_size,])
    # 第i个染色体
    for i in range(POP_size):
        pop_list = list(pop[i])
        # 第i个染色体上的基因
        # 将0去掉并变整数
        for j,each in enumerate(pop_list):
            if each == 0.0:
                index = j
                pop_list = pop_list[:j]
        for k,each in enumerate(pop_list):
            each_int = int(each)
            pop_list[k] = each_int

        fitness[i] = get_fitness(pop_list)
        print('第%d代第%d个染色体的适应度为%f'%(each_generation+1,i+1,fitness[i]))
        print('此染色体为：',pop_list)
    print('Generation:',each_generation+1,'Most fitted DNA:',pop[np.argmax(fitness),:],'适应度为：',fitness[np.argmax(fitness)])

    # 生成新的种群
    pop = select(pop,fitness)

    # 新的种群
    pop_copy = pop.copy()

    for i,parent in enumerate(pop):
        child = crossover(parent,pop_copy)
        child = mutate(child)
        pop[i] = child

