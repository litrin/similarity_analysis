import pandas as pd

from emon_analyzer import PCAClusteringDiagram

if __name__ == "__main__":
    DIMENSION = 3
    CLUSTER = 3

    DATA_FILE_NAME = "example_data.csv"
    OUTPUT_FILE_NAME = "example_onepage.pdf"

    """ Load data from an existing csv file"""
    data = pd.read_csv(DATA_FILE_NAME, index_col=0)

    """ Reduce as <DIMENSION> dimensions, distinguish as <CLUSTER> clusters"""
    diagrams = PCAClusteringDiagram(data, CLUSTER, DIMENSION)
    diagrams.analyze()

    """ save file """
    diagrams.save_pdf(OUTPUT_FILE_NAME)
