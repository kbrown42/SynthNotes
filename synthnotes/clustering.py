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
    def __init__(self, pq_files):
        self.data_dir = pq_files

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


def get_vectors(df):
    tf = TfidfVectorizer()
    return tf.fit_transform(df['sem_template'])






def template_filler(template, sentences, entities, all_mentions):
#     print(template.sem_template)
    num_start = len(entities)
    
    template_id = template.iloc[0]['sent_id']
    
    ments_in_temp = all_mentions[all_mentions.sent_id == template_id]
    
    raw_sentence = sentences[sentences.sent_id == template_id]
#     print(f'raw sent df size: {len(raw_sentence)}')
#     print(template_id)
    sent_begin = raw_sentence.iloc[0].begin
    sent_end = raw_sentence.iloc[0].end

    raw_text = raw_sentence.iloc[0].text
    
    replacements = []
#     rows_to_drop = []

#     print('Mention types in template')
#     print(ments_in_temp.mention_type.unique())
#     print('types in entities')
#     print(entities.mention_type.unique())

    for i, row in ments_in_temp.iterrows():
        ents_subset = entities[entities.mention_type == row.mention_type]

        if len(ents_subset) == 0:
            print('Empty list of doc entities')
            print(entities.mention_type)
            print(row.mention_type)
            break
        rand_ent = ents_subset.sample(n=1)
        entities = entities[entities['id'] != rand_ent.iloc[0]['id']]
#         rows_to_drop.append(rand_ent.iloc[0].name)
        
        ent_cui = rand_ent.iloc[0].cui
#         print(ent_cui)
        span_text = get_text_for_mention(ent_cui, all_mentions)
        replacements.append({
            'text' : span_text,
            'begin' : row.begin - sent_begin,
            'end' : row.end - sent_begin,
        })
        
    new_sentence = ''
    for i, r in enumerate(replacements):
        if i == 0:
            new_sentence += raw_text[0 : r['begin'] ]
        else:
            new_sentence += raw_text[replacements[i-1]['end'] : r['begin']]
        new_sentence += r['text']
    
    if(len(replacements) > 1):
        new_sentence += raw_text[replacements[-1]['end'] : ]
        
    # clean up
    num_end = len(entities)
#     print(f"Dropped {num_start - num_end} rows")
    return new_sentence, entities
    
        
        
# Find all the text associated with the cui of the mention in the template
# choose a text span based on frequency
def get_text_for_mention(cui, mentions):
    txt_counts = mentions[mentions.cui == cui].groupby('text').size().reset_index(name='cnt')
    return txt_counts.sample(n=1, weights=txt_counts.cnt).iloc[0].text




