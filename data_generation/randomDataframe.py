import numpy as np
import pandas as pd


def randomDataframe(df):
    values = {}
    for i in range(len(df.columns)):
        dist = np.random.uniform
        avg, std = df.iloc[:, i].mean(), df.iloc[:, i].std()

        # if dist == np.random.normal:
        #    values[i] = dist(avg, std, size=len(df))
        # else:
        width = np.sqrt(12) * std
        low = avg - width / 2
        high = avg + width / 2
        values[i] = dist(low, high, size=len(df))

    return pd.DataFrame(data=values)