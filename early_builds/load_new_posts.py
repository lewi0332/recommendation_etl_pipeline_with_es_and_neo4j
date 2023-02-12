from time import sleep
import json
import numpy as np
import os
import requests
from neo4j import GraphDatabase
import time
from elasticsearch import Elasticsearch
import tensorflow as tf
import boto3d
from dotenv import load_dotenv
import os

load_dotenv()

# Database Credentials
uri = os.environ.get(URI) # bolt+routing://protocol is used for the connection
userName = os.environ.get(USERNAME)
password = os.environ.get(PASSWORD)

graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

s3 = boto3.resource('s3', region_name='us-west-2')
bucket = 'prod-s3'

es = Elasticsearch(
    ["uri to elastic search"])

big_start_time = time.time()

# ------------------------------------------------------------------------------------------


def es_iterate_all_documents(es, index, pagesize=50, scroll_timeout="1d", **kwargs):
    """
    Helper to iterate ALL values from a single index
    Yields all the documents.
    """
    is_first = True
    instagram_only_clause = {"filtered": {"filter": {
        "bool": {"must": [{"term": {"provider": "instagram"}}]}}}}
    i = 0
    while True:
        # Scroll next
        if is_first:  # Initialize scroll
            body = {
                "sort": [{"published_at": "desc"}],
                "query": instagram_only_clause
            }
            result = es.search(index=index, size=pagesize,
                               scroll=scroll_timeout, **kwargs, body=body)
            is_first = False
        else:
            result = es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
        scroll_id = result["_scroll_id"]
        hits = result["hits"]["hits"]
        # Stop after no more docs
        if not hits:
            break
        # Yield each entry
        entries = [hit['_source'] for hit in hits]
        yield entries

        # time.sleep(5)
        if i > 9:
            '''This causes the loop to break at 12 rounds.'''
            break
        i += 1


# ------------------------------------------------------------------------------------------
# Image Classification portion


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def read_tensor_from_image_file(file_name, input_height=224, input_width=224, input_mean=0, input_std=255):
    with tf.Graph().as_default():
        file = s3.Object(bucket, file_name).get()['Body'].read()
        image_reader = tf.image.decode_jpeg(
            file, channels=3, name='jpeg_reader')
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(
            dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess_1 = tf.Session()
        result = sess_1.run(normalized)
        sess_1.close()
        return result


image_count = 0
labels = load_labels("./imports/retrained_labels.txt")
graph = load_graph("./imports/retrained_graph.pb")
input_layer = "input"
output_layer = "final_result"
input_name = "import/" + input_layer
output_name = "import/" + output_layer
sess = tf.Session(graph=graph)
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)


def extract_s3_url(aws_photo_url):
    return aws_photo_url.split('?')[0].split('/')[-5] + '/' + aws_photo_url.split('?')[0].split('/')[-4] \
        + '/' + aws_photo_url.split('?')[0].split('/')[-3] + '/' + aws_photo_url.split('?')[0].split('/')[-2] \
        + '/' + aws_photo_url.split('?')[0].split('/')[-1]


# Adjusted to only add to data file
data = []
round = 0
for entries in es_iterate_all_documents(es, 'posts'):
    for entry in entries:
        start = time.time()
        datum = {}
        datum['cardId'] = entry['card_id']
        try:
            key = extract_s3_url(entry['photo'])
            print("\n", key)
            t = read_tensor_from_image_file(key)
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t})
            results = np.squeeze(results)
            i = int(results.argsort()[-1:][::-1])
            image_count += 1
            print(f"\n Photo {image_count}")
            datum['cat'] = i
            print(datum)
            data.append(datum)
        except Exception as e: 
            print("oh noes")
            print(e)
            raise e
        print(f"time: {time.time()-start}")
    round += 1
    print(f"Round {round} has completed")

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"That took {time.time()-big_start_time}")

company_api = "SQL URI"
access_token = os.environ.get('ACCESS_TOKEN')
bearer_header = f'Bearer {access_token}'
headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': bearer_header
}

# fetch the last day's worth of instagram posts
url_params = {
    'count': 10000
}

print("url_params: ")
print(url_params)
response = requests.get(company_api, headers=headers, params=url_params)
results = response.json()


# params[:q] default: nil, format: "Query,Something || [Query,Something]"
# params[:platform] default: nil, format: "Platform", example: "Instagram"
# params[:filtered_platforms] default: nil, format: "Facebook,Twitter", options: Facebook, Instagram, Rss, Tumblr, Twitter, Youtube
# params[:date_from] default: 1 year ago, format: "2015-05-17" desc: "%Y-%m-%d"
# params[:date_to] default: today, format: "2015-05-17" desc: "%Y-%m-%d"
# params[:sort] default: nil, options: recent, impressions, published_at, default_returns: sorted by publised_at
# params[:page] default: 0, format: int
# params[:location] default: 0, format: int, default_returns: all areas
# params[:card] default: nil, format: int, desc: card id
# params[:exclude_cards] default: nil, format: int, desc: card id
# params[:exclude_terms] default: nil, format: string, desc: 'Worst Word'
# params[:list] default: nil, format: int, desc: list id
# params[:query_operator] default: "and", format: String, options: and, or, desc: get exactly (AND) or get post that match at least one word (OR)
# params[:size] default: 20
# params[:type] default: nil
# params[:exclude_stories] default: true
# params[:exclude_from_hidden_cards] default: nil
# params[:exclude_retweets] default: false

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

for lines in list(chunks(results, 5000)):
    with graphDB_Driver.session() as graphDB_Session:
        for line in lines:
            print(line['id'])
            graphDB_Session.run(f"MERGE (c:Card {{cardId:{line['id']}}})")

            graphDB_Session.run(f"WITH {line['tags']} as cats  MATCH (c:Card), (cat:Category) \
                                  WHERE c.cardId = {line['id']} AND cat.cat in cats \
                                  MERGE (c)-[:CONTENT_TOPIC]->(cat)")


with graphDB_Driver.session() as graphDB_Session:
    graphDB_Session.run("MATCH (cat:Category) WITH cat ORDER BY cat.cat WITH collect(cat) AS cats \
            MATCH (p:Card) \
            SET p.topic_array = algo.ml.oneHotEncoding(cats, [(p)-[:CONTENT_TOPIC]->(cat) | cat])")
