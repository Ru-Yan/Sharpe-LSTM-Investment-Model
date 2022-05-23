import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import numpy as np
import pandas as pd

dataframe = pd.read_csv('output.csv')
gold_data = dataframe.get('Gold').values[:]
len_gold_data  = len(gold_data)
bitcoin_data = dataframe.get('Bitcoin').values[:]
len_bitcoin_data  = len(bitcoin_data)
gold_bais = dataframe.get('BAIS_10_Gold').values[:]
len_gold_bais  = len(gold_bais)
bitcoin_bais = dataframe.get('BAIS_5_Bitcoin').values[:]
len_bitcoin_bais  = len(bitcoin_bais)


sum_store = 1000
sum_gold = 0
sum_bitcoin = 0