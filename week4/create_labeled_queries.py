import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

# Useful if you want to perform stemming.
import nltk
from nltk.corpus import stopwords

def cleanText(query):
    query = query.lower()
    word_tokens = query.split(" ")
 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    if len(filtered_sentence) == 0:
        return np.nan

    return " ".join(filtered_sentence)

def rollupCategory(category):
    if category not in category_counts:
        # unsure why this would happen
        print("Category not in any category counts", category)
        return category

    if category_counts[category] < min_queries:
        parent = parents_df[parents_df['category'] == category]
        if len(parent) == 0:
            # No parent or is parent
            return category
        
        return parent.iloc[0]['parent']

    print("Category greater than category counts: ", category, category_counts[category])
    return category

stemmer = nltk.stem.PorterStemmer()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# Cleaning the query
df['query'] = df['query'].apply(cleanText)

# Rolling up categories
df['category'] = df['category'].apply(rollupCategory)

# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)

"""
 import fasttext

data = pd.read_csv(output_file_name).sample(100000)

train = data.head(50000)
test = data.tail(50000)

np.savetxt(r"/workspace/datasets/query_training.txt", train.values, fmt='%s')
np.savetxt(r"/workspace/datasets/query_test.txt", test.values, fmt='%s')

model = fasttext.train_supervised(input="/workspace/datasets/query_training.txt")

model.predict("apple ipad 2 16 GB")

model.test("/workspace/datasets/query_test.txt")

model = fasttext.train_supervised(input="/workspace/datasets/query_training.txt", lr=0.5, epoch=25, wordNgrams=2)

model.test("/workspace/datasets/query_test.txt")

model.save_model("/workspace/search_with_machine_learning_course/model_query.bin")
"""