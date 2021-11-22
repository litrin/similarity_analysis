import pandas as pd
from sklearn.metrics import matthews_corrcoef


class MCCCorr:
    def __init__(self, data, main_column):
        self.data = data
        self.main_column = main_column

    def analyze(self):
        to_int = lambda a: int(a * 1000)
        main = self.data[self.main_column].apply(to_int)
        result = {}
        for key in self.data.columns:
            if key == self.main_column:
                continue
            value = self.data[key].apply(to_int)
            result[key] = matthews_corrcoef(main, value)

        return result

    def ranking(self):
        result = []
        for i, q in self.analyze().items():
            result.append((i, q))

        return sorted(result, key=lambda a: abs(a[1]), reverse=True)


if __name__ == "__main__":
    """ Load data from an existing csv file"""
    data = pd.read_csv("example_data.csv", index_col=0)

    """ Reduce as 2 dimensions, distinguish as 5 clusters"""
    diagrams = MCCCorr(data, "Backend Bound")
    print(diagrams.ranking())
