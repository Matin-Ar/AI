from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


def run(dataset):
    df = pd.read_csv(dataset)
    pca = PCA(n_components=10)
    pcs = pca.fit_transform(df.iloc[:, 8:])

    # print(pca.explained_variance_ratio_.cumsum())

    pc1_values = pcs[:, 0]
    pc2_values = pcs[:, 1]

    threshold = 0.9
    pc_values = pcs[:, 0:2]

    plt.figure(num=1, figsize=(10, 7))
    plt.title("Dendrogram")
    shc.dendrogram(shc.linkage(pc_values, method='average', metric='euclidean'))
    plt.axhline(y=threshold, color='r', linestyle='--')

    cluster = AgglomerativeClustering(
        n_clusters=None,
        metric='euclidean',
        linkage='single',
        distance_threshold=threshold
    )

    cluster.fit(pc_values)

    # find outliers
    indexes = list()
    labels = [j for j in cluster.labels_]
    for i in range(cluster.n_clusters_):
        if labels.count(i) < 3:
            for index, label in enumerate(labels):
                if label == i:
                    indexes.append(index)

    print(f'outlier_indexes in {dataset}:\n', indexes)

    plt.figure(num=2, figsize=(10, 7))
    sns.scatterplot(
        x=pc1_values,
        y=pc2_values,
        hue=cluster.labels_,
        palette="rainbow"
    ).set_title('Clusters')

    for i in indexes:
        df = df.drop(i)

    df.to_csv(f'with_PCA_output_{dataset}', encoding='utf-8', index=False)
    plt.show()


if __name__ == '__main__':
    run('female_dataset_v2.csv')
    run('male_dataset_v2.csv')