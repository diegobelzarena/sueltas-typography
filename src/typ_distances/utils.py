import pandas as pd
import numpy as np

def per_doc_inf(distances, y_names):

    distances =pd.DataFrame(distances, index=y_names, columns = y_names)
    n = len(np.unique(y_names))
    idx = np.unique(y_names, return_index=True)[1]
    names = np.array([y_names[index] for index in sorted(idx)])

    agg_distances = np.zeros((n,n))
    arg_mins = np.zeros((n,n, 2), dtype=int)
    for i in range(n-1):
        for j in range(i+1, n):
            dist = distances.loc[[names[i]], [names[j]]].values

            agg_distances[i,j] = dist.min(1).min()
            agg_distances[j,i] = dist.min(0).min()
            arg_mins[i,j] = np.unravel_index(dist.argmin(), dist.shape)
    return agg_distances, arg_mins