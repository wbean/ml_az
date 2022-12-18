"""
UCB Upper Confidence Bound 置信区间上届

针对多臂老虎机问题，在未知每个老虎机收益概率分布的情况下，获取最大收益。
探索--》利用
不断探索获取概率分布，再利用概率获取最大收益

deterministic 确定性算法
Requires update at every round 前提：即时反馈，需要实时更新上界
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../data/reinforcement_learning/Ads_CTR_Optimisation.csv")

# Implement UBC
import math

d = 10  # AD numbers 10
number_of_selections = [0] * d
sums_of_rewards = [0] * d

N = 10000  # total times
ads_selected = []
total_reword = 0
for n in range(N):
    select_ad = 0
    max_upper_bound = 0
    for ad in range(d):
        if number_of_selections[ad] > 0:
            # 平均奖励 = 总奖励除以次数
            average_reward = sums_of_rewards[ad] / number_of_selections[ad]
            delta_ad = math.sqrt(3 / 2 * math.log(n + 1) / number_of_selections[ad])
            upper_bound = average_reward + delta_ad
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            # 选取置信区间上界最大的广告
            max_upper_bound = upper_bound
            select_ad = ad
    ads_selected.append(select_ad)
    reward = dataset.values[n, select_ad]
    number_of_selections[select_ad] = number_of_selections[select_ad] + 1
    sums_of_rewards[select_ad] = sums_of_rewards[select_ad] + reward
    total_reword = total_reword + reward

# Visualising the result
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Numbers of times")
plt.show()
