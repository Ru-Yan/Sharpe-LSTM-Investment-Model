from locale import currency
import numpy as np
import math
import numba as nb

def SRp(X_g,X_b,X_c,T_g,T_b,P_g,P_b,T_c,P_c,Rf,C_g,C_b,C_c,SD):
    #期望收益率
    E = 1.5*T_g * P_g * X_g + T_b * P_b * X_b + T_c * P_c * X_c
    delta_g = abs(X_g-C_g)
    delta_c = abs(X_c-C_c)
    cost = delta_g * 0.01 + delta_c * 0.02
    ret = (E-Rf-cost)/SD
    #print('cost: ',cost,'E: ',E)
    return ret

def SD(X_g,X_b,X_c,P_g_list,P_b_list,P_c):
    Rate_Everyday = []
    for P_g,P_b in zip(P_g_list,P_b_list):
        rate = P_g * X_g + P_b * X_b + P_c * X_c
        Rate_Everyday.append(rate)
    return np.std(Rate_Everyday)

@nb.jit()
def T(avg_h,avg_i,Cv):
    Tmp = (sigmoid(avg_h * 50) + sigmoid(avg_i * 50))/(Cv * 20)
    return np.tanh(Tmp)

@nb.jit()
def sigmoid(num):
    if abs(num) > 100:
        return 0
    return 1/(1 + math.exp(-num))

@nb.jit()
def cv(data):
    mean = np.mean(data) # 平均值
    std = np.std(data, ddof=0) # 标准差 自由度
    cv = std / mean
    return cv