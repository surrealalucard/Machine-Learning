import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances) [:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence

accuracies = []

for i in range(1):
    # inputting data
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    # converting data to int list, because some of the values have '' around them, making them strings
    full_data = df.astype(float).values.tolist()
    print(full_data[:1])
    # shuffling our data
    random.shuffle(full_data)
    print(full_data[:1])
    test_size = 0.4
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    
    test_data = full_data[-int(test_size*len(full_data)):]
    
    # appending our shuffled data into our empty training dictionaries
    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    #print(train_set)
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct +=1
            total +=1

    #print('Accuracy:', correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))
