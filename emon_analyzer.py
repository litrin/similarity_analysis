from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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


class PCAClusteringDiagram:
    data = None
    cluster = 3
    kmeans = None
    pca_data = None

    _marks = list(zip(mcolors.TABLEAU_COLORS, Line2D.filled_markers))
    n_components = 0xffff

    def __init__(self, data: pd, clusters: int = 3, n_comp: int = 0xffff):
        self.data = data
        self.cluster = clusters

        if n_comp < 2:
            n_comp = 2
        self.n_components = n_comp

        if len(self.dimension) < self.n_components:
            self.n_components = len(self.dimension) - 1

    def analyze(self):
        pca = PCA(n_components=self.n_components, whiten=True)
        pca.fit(self.data)

        self.pca = pca
        self.pca_data = pca.transform(self.data)

        self.kmeans = KMeans(self.cluster)
        self.kmeans.fit(self.pca_data)

    @property
    def dimension(self):
        return self.data.columns

    @property
    def centroids(self):
        return self.kmeans.cluster_centers_

    @property
    def cluster_labels(self):
        return self.kmeans.labels_

    def _plot_chat(self, dem1, dem2):
        fig = plt.figure()
        for i, l in enumerate(self.cluster_labels):
            markers = self._marks[l]
            name = "%s: %s" % (i, self.data.index[i])
            cluster_samples = self.pca_data[i]

            plt.plot(cluster_samples[dem1], cluster_samples[dem2],
                     color=markers[0],
                     marker=markers[1], ls='None', label=name)
            plt.annotate(i,
                         xy=(cluster_samples[dem1], cluster_samples[dem2]),
                         xytext=(
                             cluster_samples[dem1], cluster_samples[dem2]),
                         fontsize=7)
        #
        # show the centroid of clusters
        # plt.scatter(self.centroids[:, dem1], self.centroids[:, dem2],
        #             color='r', marker="+")

        plt.title("PC%s + PC%s" % (dem1, dem2), fontsize=10)
        plt.xlabel("PC%s" % dem1, fontsize=8)
        plt.ylabel("PC%s" % dem2, fontsize=8)
        plt.legend(bbox_to_anchor=(1, 1), fontsize=6)

        fig.tight_layout()

    def plot_explained_variance_ratio(self, method=plt.show):
        fig, ax = plt.subplots()

        ratio_list = []
        i = 0.0
        ratio_values = self.pca.explained_variance_ratio_ * 100
        for v in ratio_values:
            i += v
            ratio_list.append(i)

        x = range(1, len(ratio_list) + 1)

        bar = ax.bar(x, ratio_values, alpha=0.5,
                     align='center', label='Individual explained variance')
        ax.bar_label(bar, label_type='edge', fmt='%.2f%%')
        ax.step(x, ratio_list, '--', color="r", alpha=0.7,
                where='mid', label='Cumulative explained variance')

        ax.set_xticks(x)
        plt.ylabel('Explained variance ratio(%)')
        plt.xlabel('Principal component index')
        ax.legend(loc='best')

        plt.tight_layout()

        yield method()
        plt.close()

    def display(self, dem1=0, dem2=1):
        self._plot_chat(dem1, dem2)
        plt.show()

    def plot_all(self, method=plt.show):
        for i in combinations(range(self.n_components), 2):
            self._plot_chat(i[0], i[1])
            yield method()
            plt.close()

    def plot_centroids_detail(self, method=plt.show):
        col = len(self.dimension)
        y_pos = range(col)

        fig, ax = plt.subplots()
        width = 1 / col  # 0.2
        offset = 0

        for i, v in enumerate(self.pca.components_):
            offset += width
            pos = [(a - 0.5) + offset for a in y_pos]

            ax.bar(pos, v, width=width * 0.95, label="PC%s" % i)

        ax.set_xticks(y_pos)
        # ax.set_xticklabels(self.dimension, fontsize=6)
        ax.set_title('Weight in Eigenvector (detail)')
        ax.legend()

        fig.tight_layout()

        yield method()
        plt.close()

    def save_pdf(self, filename):
        with PdfPages(filename) as pdf:
            for _ in self.plot_all(pdf.savefig):
                pass

            for _ in self.plot_explained_variance_ratio(pdf.savefig):
                pass

            # for _ in self.plot_centroids_detail(pdf.savefig):
            #     pass


class WorkloadSimilarityAnalyzer:

    def __init__(self, filename):
        self.data = pd.read_csv(filename, index_col=0)

    def raw(self) -> pd.DataFrame:
        return self.data

    def pca(self, n_components=14) -> pd.DataFrame:
        pca = PCA(n_components=n_components, copy=True)
        pca.fit(self.data)
        pca_data = pca.transform(self.data)
        print(pca.explained_variance_ratio_)
        pca_data = pd.DataFrame(pca_data)
        pca_data.index = self.data.index

        return pca_data

    def cov(self) -> pd.DataFrame:
        cov = np.cov(self.data)

        cov_data = pd.DataFrame(cov)
        cov_data.index = self.data.index

        return cov_data

    def get_matrix(self, data, algorithm="euclidean",
                   rate=False) -> pd.DataFrame:
        dist_raw = pdist(data, metric=algorithm)
        dis_matrix = squareform(dist_raw)

        df = pd.DataFrame(dis_matrix)
        df.index = self.data.index
        df.columns = self.data.index

        if not rate:
            return df
        return 1 - df / df.values.max()
