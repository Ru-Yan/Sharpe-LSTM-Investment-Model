import math
import random
from pickle import FALSE, TRUE
import Srp
import pandas as pd
from pandas import DataFrame
import numba as nb
import numpy as np

P_c_day = 0.0016


class Population:
    # 种群的设计
    def __init__(self, size, chrom_size, cp, mp, gen_max):
        # 种群信息合
        self.individuals = np.array([])       # 个体集合
        self.individuals.shape = 0,3
        self.fitness = np.array([])               # 个体适应度集
        self.selector_probability = np.array([])   # 个体选择概率集合
        self.new_individuals = np.array([])      # 新一代个体集合
        self.new_individuals.shape = 0,3
        self.old_individuals = np.array([]) 
        self.old_individuals.shape = 0,3
        self.max_data = np.float(-200)
        self.max_x = np.float(0)
        self.max_y = np.float(0)
        self.max_z = np.float(0)

        self.elitist = {'chromosome': [0, 0, 0],
                        'fitness': 0, 'age': 0}  # 最佳个体的信息

        self.size = np.int(size)  # 种群所包含的个体数
        self.chromosome_size = np.int(chrom_size)  # 个体的染色体长度
        self.crossover_probability = np.float(cp)   # 个体之间的交叉概率
        self.mutation_probability = np.float(mp)    # 个体之间的变异概率

        self.generation_max = np.int(gen_max)  # 种群进化的最大世代数
        self.age = np.int(0)                  # 种群当前所处世代

        # 随机产生初始个体集，并将新一代个体、适应度、选择概率等集合以 0 值进行初始化
        v = np.power(2,self.chromosome_size - 1)
        for i in range(self.size):
            self.individuals = np.append(self.individuals,[[random.randint(0, v), random.randint(0, v), random.randint(0, v)]],0)
            self.new_individuals = np.append(self.new_individuals,[[0, 0, 0]],0)
            self.old_individuals = np.append(self.old_individuals,[[0, 0, 0]],0)
            self.fitness = np.append(self.fitness,0)
            self.selector_probability = np.append(self.selector_probability,0)
    
    # 基于轮盘赌博机的选择
    def decode(self, interval, chromosome):
        '''将一个染色体 chromosome 映射为区间 interval 之内的数值'''
        d = interval[1] - interval[0]
        n = float(np.power(2,self.chromosome_size - 1))
        return (interval[0] + chromosome * d / n)

    def fitness_func(self, chrom1, chrom2, chrom3, T_g, T_b, P_g, P_b, T_c, P_c, Rf, C_g, C_b, C_c, P_g_list, P_b_list):
        '''适应度函数，可以根据个体的两个染色体计算出该个体的适应度'''
        interval = [0.0, 1.0]
        (X_g, X_b, X_c) = (self.decode(interval, chrom1),
                           self.decode(interval, chrom2),
                           self.decode(interval, chrom3))
        #####################################
        if(X_g+X_b+X_c < 0.995 or X_g+X_b+X_c > 1.005):
            return -100
            # return 0.001 / (math.sqrt((X_g+X_b+X_c-1)**2))
        X_g = np.float(X_g)
        X_b = np.float(X_b)
        X_c = np.float(X_c)
        P_g_list = np.array(P_g_list)
        P_b_list = np.array(P_b_list)
        P_c = np.float(P_c)
        SD = Srp.SD(X_g, X_b, X_c, P_g_list, P_b_list, P_c)
        func = Srp.SRp(X_g, X_b, X_c, T_g, T_b, P_g, P_b,
                       T_c, P_c, Rf, C_g, C_b, C_c, SD)
        return func
        # return X_g+2*X_b+3*X_c

    def evaluate(self, T_g, T_b, P_g, P_b, T_c, P_c, Rf, C_g, C_b, C_c, P_g_list, P_b_list):
        '''用于评估种群中的个体集合 self.individuals 中各个个体的适应度'''
        sp = self.selector_probability
        for i in range(self.size):
            self.fitness[i] = self.fitness_func(self.individuals[i][0],   # 将计算结果保存在 self.fitness 列表中
                                                self.individuals[i][1],
                                                self.individuals[i][2],
                                                T_g, T_b, P_g, P_b, T_c, P_c, Rf, C_g, C_b, C_c, P_g_list, P_b_list)
        ft_sum = sum(self.fitness)
        for i in range(self.size):
            sp[i] = self.fitness[i] / float(ft_sum)   # 得到各个个体的生存概率
        for i in range(1, self.size):
            sp[i] = sp[i] + sp[i-1]   # 需要将个体的生存概率进行叠加，从而计算出各个个体的选择概率

    # 轮盘赌博机（选择）
    def select(self):
        (t, i) = (random.random(), 0)
        for p in self.selector_probability:
            if p > t:
                break
            i = i + 1
        if(i == len(self.selector_probability)):
            i = i-1
        return i
    
    # 交叉
    def cross(self, chrom1, chrom2):
        p = random.random()    # 随机概率
        n = np.power(2,self.chromosome_size - 1)
        if chrom1 != chrom2 and p < self.crossover_probability:
            t = random.randint(1, self.chromosome_size - 1)   # 随机选择一点（单点交叉）
            mask = n << t    # << 左移运算符
            # & 按位与运算符：参与运算的两个值,如果两个相应位都为1,则该位的结果为1,否则为0
            a = int(mask)
            chrom1 = int(chrom1)
            chrom2 = int(chrom2)
            (r1, r2) = (chrom1 & a, chrom2 & a)
            mask = n >> (self.chromosome_size - t)
            (l1, l2) = (chrom1 & mask, chrom2 & mask)
            (chrom1, chrom2) = (r1 + l2, r2 + l1)
        return (chrom1, chrom2)

    # 变异
    def mutate(self, chrom):
        p = random.random()
        if p < self.mutation_probability:
            t = random.randint(1, self.chromosome_size)
            mask1 = 1 << (t - 1)
            mask1 = int(mask1)
            chrom = int(chrom)
            mask2 = chrom & mask1
            if mask2 > 0:
                chrom = chrom & (~mask2)  # ~ 按位取反运算符：对数据的每个二进制位取反,即把1变为0,把0变为1
            else:
                chrom = chrom ^ mask1   # ^ 按位异或运算符：当两对应的二进位相异时，结果为1
        return chrom

    # 保留最佳个体
    def reproduct_elitist(self):
        # 与当前种群进行适应度比较，更新最佳个体
        j = -1
        for i in range(self.size):
            if self.elitist['fitness'] < self.fitness[i]:
                j = i
                self.elitist['fitness'] = self.fitness[i]
        if (j >= 0):
            self.elitist['chromosome'][0] = self.individuals[j][0]
            self.elitist['chromosome'][1] = self.individuals[j][1]
            self.elitist['chromosome'][2] = self.individuals[j][2]
            self.elitist['age'] = self.age

    # 进化过程
    def evolve(self, T_g, T_b, P_g, P_b, T_c, P_c, Rf, C_g, C_b, C_c, P_g_list, P_b_list):
        indvs = self.individuals
        new_indvs = self.new_individuals
        # 计算适应度及选择概率
        self.evaluate(T_g, T_b, P_g, P_b, T_c, P_c, Rf,
                      C_g, C_b, C_c, P_g_list, P_b_list)
        # 进化操作
        i = 0
        while True:
            # 选择两个个体，进行交叉与变异，产生新的种群
            idv1 = self.select()
            idv2 = self.select()
            # 交叉
            (idv1_x, idv1_y) = (indvs[idv1][0], indvs[idv1][1])
            (idv2_x, idv2_y) = (indvs[idv2][0], indvs[idv2][1])
            (idv1_x, idv1_z) = (indvs[idv1][0], indvs[idv1][2])
            (idv2_x, idv2_z) = (indvs[idv2][0], indvs[idv2][2])
            (idv1_y, idv1_z) = (indvs[idv1][1], indvs[idv1][2])
            (idv2_y, idv2_z) = (indvs[idv2][1], indvs[idv2][2])
            (idv1_x, idv2_x) = self.cross(idv1_x, idv2_x)
            (idv1_y, idv2_y) = self.cross(idv1_y, idv2_y)
            (idv1_z, idv2_z) = self.cross(idv1_z, idv2_z)
            # 变异
            (idv1_x, idv1_y, idv1_z) = (self.mutate(idv1_x),
                                        self.mutate(idv1_y), self.mutate(idv1_z))
            (idv2_x, idv2_y, idv2_z) = (self.mutate(idv2_x),
                                        self.mutate(idv2_y), self.mutate(idv2_z))
            (new_indvs[i][0], new_indvs[i][1], new_indvs[i][2]) = (
                idv1_x, idv1_y, idv1_z)  # 将计算结果保存于新的个体集合self.new_individuals中
            (new_indvs[i+1][0], new_indvs[i+1][1],
             new_indvs[i+1][2]) = (idv2_x, idv2_y, idv2_z)
            # 判断进化过程是否结束
            i = i + 2         # 循环self.size/2次，每次从self.individuals 中选出2个
            if i >= self.size:
                break

        # 最佳个体保留
        # 如果在选择之前保留当前最佳个体，最终能收敛到全局最优解。
        self.reproduct_elitist()

        # 更新换代：用种群进化生成的新个体集合 self.new_individuals 替换当前个体集合
        for i in range(self.size):
            self.old_individuals[i][0] = self.individuals[i][0]
            self.old_individuals[i][1] = self.individuals[i][1]
            self.old_individuals[i][2] = self.individuals[i][2]

        for i in range(self.size):
            self.individuals[i][0] = self.new_individuals[i][0]
            self.individuals[i][1] = self.new_individuals[i][1]
            self.individuals[i][2] = self.new_individuals[i][2]
            
    def run(self, T_g, T_b, P_g, P_b, T_c, P_c, Rf, C_g, C_b, C_c, P_g_list, P_b_list):
        '''根据种群最大进化世代数设定了一个循环。
        在循环过程中，调用 evolve 函数进行种群进化计算，并输出种群的每一代的个体适应度最大值、平均值和最小值。'''
        interval = [0.0, 1.0]
        for i in range(self.generation_max):
            self.evolve(T_g, T_b, P_g, P_b, T_c, P_c, Rf,
                        C_g, C_b, C_c, P_g_list, P_b_list)
            if(self.max_data < max(self.fitness)):
                self.max_data = max(self.fitness)
                max_index = np.argwhere(self.fitness == max(self.fitness))
                max_individual = self.old_individuals[max_index]
                self.max_x = self.decode(
                    interval, max_individual[0][0][0])
                self.max_y = self.decode(
                    interval, max_individual[0][0][1])
                self.max_z = self.decode(
                    interval, max_individual[0][0][2])
            #print('进化次数： ', i, ' 个体适应度最大值', max(self.fitness), ' 历史最大值： ', self.max_data,
             #     ' 对应的g： ', self.max_x, ' 对应的b： ', self.max_y, ' 对应的c： ', self.max_z)
        return self.max_data, self.max_x, self.max_y, self.max_z

@nb.jit()
def ifChange(row):
    # 乖离率超标
    if abs(row["BAIS_5_Bitcoin"]) > 0.05 or abs(row["BAIS_10_Gold"]) > 0.1:
        return True
    # 后天为休息日，明天要操作
    if INPUT.iloc[index+2, 4] == FALSE and INPUT.iloc[index, 4] == TRUE:
        return True
    # 风险剧烈变动
    if abs(INPUT.iloc[index, 22]-INPUT.iloc[index-1, 22]) > 0.2 or abs(INPUT.iloc[index, 23]-INPUT.iloc[index-1, 23]) > 0.2:
        return True
    return False

@nb.jit()
def UpdateValue(row):
    Bitcoin_Today = INPUT.iloc[index-1, 16]*row["Bitcoin"]
    Gold_Today = INPUT.iloc[index-1, 18]*row["Gold"]
    Cash_Today = INPUT.iloc[index-1, 19]*P_c_day
    New_Jingzhi = Bitcoin_Today + Gold_Today + Cash_Today

    New_Shouyi_B = (Bitcoin_Today - 333.33)/333.33
    New_Shouyi_G = (Gold_Today - 333.33)/333.33

    New_Fengxian_B = Srp.T(
        New_Shouyi_B, row["30日平均涨跌 比特币"], row["30日变异系数 Bitcoin"])
    New_Fengxian_G = Srp.T(
        New_Shouyi_G, row["30日平均涨跌 黄金"], row["30日变异系数 Gold"])

@nb.jit()
def Change(row):
    T_g = row["黄金风险系数"]
    T_b = row["比特币风险系数"]
    print(INPUT.iloc[index, 3])
    P_g = (INPUT.iloc[index+5, 8]-INPUT.iloc[index, 3])/INPUT.iloc[index, 3]
    P_b = (INPUT.iloc[index+5, 7]-INPUT.iloc[index, 2])/INPUT.iloc[index, 2]
    T_c = 1
    P_c = P_c_day*5
    Rf = P_c
    C_g = 0.333
    C_b = 0.333
    C_c = 0.334

    P_g_list = []
    P_g_list.append(
        (INPUT.iloc[index, 8]-INPUT.iloc[index-1, 3])/INPUT.iloc[index, 3])
    for i in range(1, 5):
        P_g_list.append(
            (INPUT.iloc[index+i, 8]-INPUT.iloc[index+i-1, 8])/INPUT.iloc[index, 8])

    P_b_list = []
    P_b_list.append(
        (INPUT.iloc[index, 7]-INPUT.iloc[index-1, 2])/INPUT.iloc[index, 2])
    for i in range(1, 5):
        P_b_list.append(
            (INPUT.iloc[index+i, 7]-INPUT.iloc[index+i-1, 7])/INPUT.iloc[index, 7])

    pop = Population(500, 20, 0.4, 0.5, 800)
    max_srp, max_g, max_b, max_c = pop.run(
        T_g, T_b, P_g, P_b, T_c, P_c, Rf, C_g, C_b, C_c, P_g_list, P_b_list)
    OUTPUT = OUTPUT.append({'index': index, 'S': max_srp,
                            'G': max_g,
                            'B': max_b,
                            'C': max_c}, ignore_index=True)


if __name__ == '__main__':
    # 种群的个体数量为 50，染色体长度为 25，交叉概率为 0.8，变异概率为 0.1,进化最大世代数为 150
    INPUT = pd.read_csv("./Output_Test.csv")
    OUTPUT = DataFrame(columns=('index', 'S', 'G', 'B', 'C'))

    for index, row in INPUT.iterrows():
        if index < 8 or index >= 1820:
            continue

        # 更新当天净值########################
        Bitcoin_Today = INPUT.iloc[index-1, 16]*row["Bitcoin"]
        Gold_Today = INPUT.iloc[index-1, 18]*row["Gold"]
        Cash_Today = INPUT.iloc[index-1, 19]*(1+P_c_day)
        New_Jingzhi = Bitcoin_Today + Gold_Today + Cash_Today

        Bitcoin_7d_ago = INPUT.iloc[index-7, 16]*INPUT.loc[index-7, "Bitcoin"]
        Gold_7d_ago = INPUT.iloc[index-7, 18]*INPUT.loc[index-7, "Gold"]
#######################222#####################
        New_Shouyi_B = (Bitcoin_Today - 333.33)/333.33
        New_Shouyi_G = (Gold_Today - 333.33)/333.33

        print("!index:",index)
        New_Fengxian_B = Srp.T(
            New_Shouyi_B, row["30日平均涨跌 比特币"], row["30日变异系数 Bitcoin"])
        New_Fengxian_G = Srp.T(
            New_Shouyi_G, row["30日平均涨跌 黄金"], row["30日变异系数 Gold"])

        INPUT.loc[index, "当日比特币价值"] = Bitcoin_Today
        INPUT.loc[index, "当日黄金价值"] = Gold_Today
        INPUT.loc[index, "当日现金价值"] = Cash_Today
        INPUT.loc[index, "当日收盘净值"] = New_Jingzhi
        INPUT.loc[index, "黄金收益率"] = New_Shouyi_G
        INPUT.loc[index, "比特币收益率"] = New_Shouyi_B
        INPUT.loc[index, "比特币风险系数"] = New_Fengxian_B
        INPUT.loc[index, "黄金风险系数"] = New_Fengxian_G
        #####################################

        #判断第二天是否交易#####################
        Trade = False
        # 乖离率超标
        if abs(row["BAIS_5_Bitcoin"]) > 0.04 or abs(row["BAIS_10_Gold"]) > 0.08:
            Trade = True
        # 后天为休息日，明天要操作
        if INPUT.iloc[index+2, 4] == FALSE and INPUT.iloc[index+1, 4] == TRUE:
            Trade = True
        # 风险剧烈变动
        if abs(INPUT.iloc[index, 22]-INPUT.iloc[index-1, 22]) > 0.2 or abs(INPUT.iloc[index, 23]-INPUT.iloc[index-1, 23]) > 0.2:
            Trade = True

        # 若明天是非交易日，则无论以上何种情况，均不交易
        if INPUT.iloc[index+1, 4] == FALSE:
            Trade = False
        ######################################

        #计算新的比例##########################
        if Trade == False:
            # 不交易，明天的份额和今天的一样
            INPUT.iloc[index, 16] = INPUT.iloc[index-1, 16]
            INPUT.iloc[index, 18] = INPUT.iloc[index-1, 18]
            INPUT.iloc[index, 19] = Cash_Today
        else:
            P_g = (INPUT.iloc[index+5, 8] -
                   INPUT.iloc[index, 3])/INPUT.iloc[index, 3]
            P_b = (INPUT.iloc[index+5, 7] -
                   INPUT.iloc[index, 2])/INPUT.iloc[index, 2]
            P_c = P_c_day*5
            Rf = P_c
            C_g = Gold_Today/New_Jingzhi
            C_b = Bitcoin_Today/New_Jingzhi
            C_c = Cash_Today/New_Jingzhi

            # 预测黄金未来五天每天的收益率
            P_g_list = []
            P_g_list.append(
                (INPUT.iloc[index, 8]-INPUT.iloc[index-1, 3])/INPUT.iloc[index, 3])
            for i in range(1, 5):
                P_g_list.append(
                    (INPUT.iloc[index+i, 8]-INPUT.iloc[index+i-1, 8])/INPUT.iloc[index, 8])

            # 预测比特币未来五天每天的收益率
            P_b_list = []
            P_b_list.append(
                (INPUT.iloc[index, 7]-INPUT.iloc[index-1, 2])/INPUT.iloc[index, 2])
            for i in range(1, 5):
                P_b_list.append(
                    (INPUT.iloc[index+i, 7]-INPUT.iloc[index+i-1, 7])/INPUT.iloc[index, 7])
###
            pop = Population(150, 20, 0.4, 0.5, 100)
            max_srp, max_g, max_b, max_c = pop.run(
                New_Fengxian_G, New_Fengxian_B, P_g, P_b, 1, P_c, Rf, C_g, C_b, C_c, P_g_list, P_b_list)
            
            max_sum = max_g + max_b + max_c
            max_g = round(max_g / max_sum,6)
            max_b = round(max_b / max_sum,6)
            max_c = round(max_c / max_sum,6)
            
            #计算新的份额
            print({'index': index, 'S': max_srp, 'G': max_g,'B': max_b,'C': max_c})
            INPUT.loc[index, "夏普比率"] = max_srp
            INPUT.loc[index, "G"] = max_g
            INPUT.loc[index, "B"] = max_b
            INPUT.loc[index, "C"] = max_c
            #########111##############################################################################
            if max_srp <= 0:
                # 夏普指数为负数，不交易，明天的份额和今天的一样
                INPUT.iloc[index, 16] = INPUT.iloc[index-1, 16]
                INPUT.iloc[index, 18] = INPUT.iloc[index-1, 18]
                INPUT.iloc[index, 19] = Cash_Today
            else:
                # 交易，计算出新的份额

                #新的目标价值
                New_Gold = round(New_Jingzhi * max_g,2)
                New_Bitcoin = round(New_Jingzhi * max_b,2)
                #New_Cash = New_Jingzhi - New_Gold - New_Bitcoin

                #交易发生的手续费
                Gold_fee = abs(New_Gold - Gold_Today)*0.01
                Bitcoin_fee = abs(New_Bitcoin - Bitcoin_Today)*0.02
                INPUT.loc[index, "预扣除手续费"] = Gold_fee + Bitcoin_fee
                New_Jingzhi = New_Jingzhi - Gold_fee - Bitcoin_fee

                #计算修正净值后的比例
                New_Gold = round(New_Jingzhi * max_g,2)
                New_Bitcoin = round(New_Jingzhi * max_b,2)
                New_Cash = round(New_Jingzhi - New_Gold - New_Bitcoin,2)

                #新的份额
                INPUT.iloc[index, 16] = New_Bitcoin/row["Bitcoin"]
                INPUT.iloc[index, 18] = New_Gold/row["Gold"]
                INPUT.iloc[index, 19] = New_Cash
        ####################################################
    INPUT.to_excel("./Part_Output.xlsx")
