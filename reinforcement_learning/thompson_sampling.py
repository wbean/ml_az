"""
汤普森抽样算法

预测奖励的平均值（数学期望）而非预测概率分布

probabilistic 随机性算法
Can accommodate delayed feedback 允许延迟更新、批量更新
Better empirical evidence 更好的实际应用效果
"""
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../data/reinforcement_learning/Ads_CTR_Optimisation.csv")

# Implement Thompson Sampling
import random

d = 10  # AD numbers 10
number_of_reward0 = [0] * d
number_of_reward1 = [0] * d

N = 10000  # total times
ads_selected = []
total_reword = 0
for n in range(N):
    select_ad = 0
    max_random = 0
    for ad in range(d):
        random_beta = random.betavariate(number_of_reward1[ad] + 1, number_of_reward0[ad] + 1)
        if random_beta > max_random:
            max_random = random_beta
            select_ad = ad
    ads_selected.append(select_ad)
    reward = dataset.values[n, select_ad]
    if reward == 1:
        number_of_reward1[select_ad] = number_of_reward1[select_ad] + 1
    else:
        number_of_reward0[select_ad] = number_of_reward0[select_ad] + 1
    total_reword = total_reword + reward

# Visualising the result
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Numbers of times")
plt.show()
