"""
support(M1) = M1 / ALL #M1的支持度
support(M2) = M2 / ALL #M2的支持度
confidence(M1->M2) = M1 && M2 / M1 #M1对M2信心水准
lift(M1->M2) =  confidence(M1->M2) / support(M2) #M1对M2的提升度

steps
1: set a minimum support and confidence
2: take all the subsets in transactions having higher support than minimum support
3: take all the rules of these subsets having higher confidence than minimum confidence
4: sort the rules by decreasing lift
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

start = time.perf_counter()


def get_transaction(values):
    t = []
    for i in range(0, 7500):
        t.append([str(values[i, j]) for j in range(0, 20)])
    return t


dataset = pd.read_csv('../data/association_rule_learning/Market_Basket_Optimisation.csv')
transactions = get_transaction(dataset.values)

from apyori import apriori

rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

results = list(rules)
myResults = [list(x) for x in results]
end = time.perf_counter()
print(end - start)
