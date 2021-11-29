import pandas as pd

from emon_analyzer import MCCCorr

if __name__ == "__main__":
    """ Load data from an existing csv file"""
    data = pd.read_csv("example_data.csv", index_col=0)

    """ Reduce as 2 dimensions, distinguish as 5 clusters"""
    diagrams = MCCCorr(data, "Backend Bound")
    print(diagrams.ranking())
