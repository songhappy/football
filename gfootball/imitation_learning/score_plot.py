import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input1 ="/Users/guoqiong/intelWork/projects/googleFootball/documents/trained100.txt"
input2 ="/Users/guoqiong/intelWork/projects/googleFootball/documents/agent100.txt"
def get_scores(input_file):
    with open(input_file) as f:
        X = [line.strip() for line in f]
    scores = []
    diffs = []
    for x in X:
        s = re.findall(r"[-+]?\d*\.\d+|\d+", x)
        s = [int(ss) for ss in s]
        diff = s[-2] - s[-1]
        s.append(diff)
        scores.append(s)
        diffs.append(diff)
    df = pd.DataFrame(scores, columns =['episode', 'score1', 'score2', 'diff'])
#print(df)
    statsScores = df.describe()
    print(statsScores)
    stats = df[['diff','episode']].groupby('diff').agg('count').reset_index()
    return stats

stats1 = get_scores(input1).rename(columns = {'episode':'google'})
stats2 = get_scores(input2).rename(columns = {'episode':'IL'})#.set_index("diff")
stats = pd.merge(stats1, stats2, on="diff",how='outer').sort_values("diff")
print(stats)
stats.plot.bar(x='diff', y=['google', 'IL'], rot=0)
plt.savefig("/Users/guoqiong/intelWork/projects/googleFootball/documents/trainedVsAgent.png")
plt.show()
