import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastparquet import ParquetFile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, MiniBatchKMeans


class Clusterer(object):
    def __init__(self, pq_files, output):
        self.data_dir = pq_files
        self.output = output

    def cluster(self, n_clusters=120):
        templates = pd.read_parquet(f'{self.data_dir}/templates.parquet' )
        sentences = pd.read_parquet(f'{self.data_dir}/sentences.parquet')
        mentions = pd.read_parquet(f'{self.data_dir}/mentions.parquet')
        umls = pd.read_parquet(f'{self.data_dir}/umls.parquet')

        km = KMeans( init='k-means++', max_iter=100, n_init=1,
                 n_clusters=n_clusters, verbose=False)

        vectors = get_vectors(templates)

        km.fit(vectors)
        predictions = km.predict(vectors)
        sil_score = silhouette_score(vectors, predictions, metric='euclidean')

        templates['cluster'] = predictions

        sentences = sentences.merge(templates[['sent_id', 'cluster']], on='sent_id')
        mentions = mentions.merge(templates[['sent_id', 'cluster']], on='sent_id')

        templates.to_parquet(f'{self.output}/templates.parquet')
        sentences.to_parquet(f'{self.output}/sentences.parquet')
        mentions.to_parquet(f'{self.output}/mentions.parquet')


def get_vectors(df):
    tf = TfidfVectorizer()
    return tf.fit_transform(df['sem_template'])








