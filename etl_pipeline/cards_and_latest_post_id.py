"""
This script is used to refresh the cards and latest post id in neo4j.
It is run every week to extract the latest data from elasticsearch
and update the neo4j database with images classification classifications
and update the document vectors for each card_id.

Author: Derrick Lewis
Date: 2019-08-01
"""
import time
import os
from neo4j import GraphDatabase
from elasticsearch import Elasticsearch
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
import spacy
import numpy as np
import tensorflow as tf
import boto3
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()

# Database Credentials
uri = os.environ.get(URI) # bolt+routing://protocol is used for the connection
userName = os.environ.get(USERNAME)
password = os.environ.get(PASSWORD)

# Connect to the neo4j database server
graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

TF_pb = os.environ.get(GRAPH_PB) # Path to the tf1.0 graph.pb file

es = Elasticsearch(["qbox instance of elasticsearch"])

s3 = boto3.resource("s3", region_name="us-west-2")
bucket = "prod-s3"

nltk.download("stopwords")

"""Move the 11gb file model.txt to the CWD and run this in terminal"""
# !python -m spacy init-model en /tmp/custom_vectors --vectors-loc /home/ec2-user/graph/refresh/model2.txt

"""Loading dictionary of words into Spacy"""
nlp = spacy.load("/custom_vectors")

def es_iterate_all_documents(
    es, index, query=False, pagesize=10000, scroll_timeout="1m", **kwargs
):
    """
    Helper to iterate ALL values from a single index
    Yields all the documents.
    """
    is_first = True
    while True:
        # Scroll next
        if is_first:  # Initialize scroll
            body = {"size": pagesize}
            if query:
                body["query"] = query
            result = es.search(index=index, scroll="1m", **kwargs, body=body)
            is_first = False
        else:
            body = {"scroll_id": scroll_id, "scroll": scroll_timeout}
            result = es.scroll(body=body)
        scroll_id = result["_scroll_id"]
        hits = result["hits"]["hits"]
        # Stop after no more docs
        if not hits:
            break
        # Yield each entry
        yield from (hit["_source"] for hit in hits)

        time.sleep(1)



# Pull all the card ids from elasticsearch
cards_with_instagram = {
    "filtered": {
        "filter": {
            "bool": {
                "must_not": [{"term": {"instagram": 0}}],
                "must": [{"exists": {"field": "instagram"}}],
            }
        }
    }
}

all_es_card_ids = []
for card in es_iterate_all_documents(es, "cards", cards_with_instagram):
    all_es_card_ids.append(
        (card["id"], card["tags"], card["instagram"], card["instagram_engagement"])
    )


# Pull all the current card ids from neo4j
with graphDB_Driver.session() as graphDB_Session:
    temp = graphDB_Session.run(f"MATCH (c:Card) RETURN c.cardId")
    all_neo_card_ids = [r.values()[0] for r in temp]


# ---------------------------------------------------------------------
# Get a list of any new card ids not yet in neo4j

def diff(first, second):
    """
    Get a list of any new card ids not yet in neo4j

    Parameters
    ---
    first: list
        list of card ids from elasticsearch
    second: list
        list of card ids from neo4j

    Returns
    ---
    list
        list of card ids not yet in neo4j
    """
    second = set(second)
    return [item for item in first if item[0] not in second]

new_card_ids = diff(all_es_card_ids, all_neo_card_ids)
new_fol_card_ids = []
for val in new_card_ids:
    if val[2] is not None and val[3] is not None:
        new_fol_card_ids.append(val)

new_fol_card_ids_only = [line[0] for line in new_fol_card_ids]


def fetch_posts_for(card_id):
    """INPUT: card ID as int.
       Elastic search query to return instagram post data for the last 33 posts"""
    instagram_only_clause = {
        "filtered": {
            "filter": {
                "bool": {
                    "must": [
                        {"term": {"provider": "instagram"}},
                        {"term": {"card_id": card_id}},
                    ],
                    "must_not": [{"prefix": {"type": "story"}}],
                }
            }
        }
    }
    body = {"sort": [{"published_at": "desc"}], "query": instagram_only_clause}
    result = es.search(index="posts", size=33, body=body)
    hits = result["hits"]["hits"]
    return [x["_source"] for x in hits]


# ---------------------------------------------------------------------
# SYPHER Ctuff

# Add cards and the relationships to category nodes
count = 0
with graphDB_Driver.session() as graphDB_Session:
    for line in new_fol_card_ids:
        print(line[0])
        graphDB_Session.run(
            f"MERGE (c:Card {{cardId:{line[0]}}}) \
            ON MATCH SET c.igFollower = {line[2]}, c.igEngagement = {line[3]} \
            ON CREATE SET c.igFollower = {line[2]}, c.igEngagement = {line[3]}"
        )

        graphDB_Session.run(
            f"WITH {line[1]} as cats  MATCH (c:Card), (cat:Category) \
                              WHERE c.cardId = {line[0]} AND cat.cat in cats \
                              MERGE (c)-[:CONTENT_TOPIC]->(cat)"
        )
        count += 1
        if count % 10000 == 0:
            graphDB_Session.sync()

# Use the new relationships to update the Category vectors.
with graphDB_Driver.session() as graphDB_Session:
    graphDB_Session.run(
        "MATCH (cat:Category) WITH cat ORDER BY cat.cat WITH collect(cat) AS cats \
            MATCH (p:Card) \
            SET p.topic_array = algo.ml.oneHotEncoding(cats, [(p)-[:CONTENT_TOPIC]->(cat) | cat])"
    )


# ---------------------------------------------------------------------
# Set up Tensorflow inference functions

def load_TF_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def load_readable_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def read_tensor_from_image_file(
    file_name, input_height=224, input_width=224, input_mean=0, input_std=255
):
    """
    Reads a jpeg file from AWS and returns a tensor suitable for TF inference

    Parameters
    ---
    file_name: str
        name of the file in AWS S3
    input_height: int
        height of the image
    input_width: int
        width of the image
    input_mean: int
        mean of the image
    input_std: int
        standard deviation of the image

    Returns
    ---
    tensor
        tensor suitable for TF inference

    """
    with tf.Graph().as_default():
        # Using context manager to close session and avoid memory leaks
        file = s3.Object(bucket, file_name).get()["Body"].read()
        image_reader = tf.image.decode_jpeg(file, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess_1 = tf.Session()
        result = sess_1.run(normalized)
        sess_1.close()
        return result


def extract_s3_url(aws_photo_url):
    return (
        aws_photo_url.split("?")[0].split("/")[-5]
        + "/"
        + aws_photo_url.split("?")[0].split("/")[-4]
        + "/"
        + aws_photo_url.split("?")[0].split("/")[-3]
        + "/"
        + aws_photo_url.split("?")[0].split("/")[-2]
        + "/"
        + aws_photo_url.split("?")[0].split("/")[-1]
    )


def categorize_image(entry):
    try:
        datum = {}
        datum["cardId"] = entry["card_id"]
        key = extract_s3_url(entry["photo"])
        t = read_tensor_from_image_file(key)
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
        results = np.squeeze(results)
        i = int(results.argsort()[-1:][::-1])
        datum["cat"] = i
        print(f"processing {entry['card_id']}")
        return datum
    except Exception as err:
        print("something went wrong during photo comprehension")
        print(err)


image_count = 0

labels = load_readable_labels("../imports/retrained_labels.txt")

graph = load_TF_graph(TF_pb)
input_layer = "input"
output_layer = "final_result"
input_name = "import/" + input_layer
output_name = "import/" + output_layer
sess = tf.Session(graph=graph)
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

# ---------------------------------------------------------------------
# Run the inference on the new posts

data = []
round = 0
for card_id in new_fol_card_ids:
    card_id = card_id[0]
    posts = fetch_posts_for(card_id)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        threads = executor.map(categorize_image, posts)
        for _ in threads:
            if _ is None:
                pass
            else:
                data.append(_)
    round += 1
    print(f"Round {round} of {len(new_fol_card_ids)} posts created")


# ---------------------------------------------------------------------
# Update Neo4j Graph card node to increase the image count:

image_data = []
for val in data:
    if val is not None:
        image_data.append(val)

count = 0
with graphDB_Driver.session() as graphDB_Session:
    for i in image_data:
        graphDB_Session.run(
            f"MATCH (c:Card {{cardId:{i['cardId']}}}), (img:Imagetype {{cat:{i['cat']}}}) \
        MERGE (c)-[r:IMAGE_TOPIC]->(img) \
        ON CREATE SET r.count = 1 \
        ON MATCH SET r.count = r.count + 1"
        )
        count += 1
        if count % 5000 == 0:
            graphDB_Session.sync()


# Find the absent relationships and set count to zero
with graphDB_Driver.session() as graphDB_Session:
    graphDB_Session.run(
        f"CALL apoc.periodic.iterate('MATCH (c:Card), (img:Imagetype) WHERE c.cardId \
        in {new_fol_card_ids_only} RETURN c, img', \
        'MERGE (c)-[r:IMAGE_TOPIC]->(img) ON CREATE SET r.count = 0', {{batchSize:5000}})"
    )

# Collect counts from all relationships and build image vector
with graphDB_Driver.session() as graphDB_Session:
    graphDB_Session.run(
        f"MATCH (c:Card), (img:Imagetype) WHERE c.cardId \
        in {new_fol_card_ids_only} OPTIONAL MATCH (c)-[rel:IMAGE_TOPIC]->(img) \
        WITH {{id:c.cardId, weights: collect(coalesce(rel.count, algo.NaN()))}} as userData \
        WITH collect(userData) as data UNWIND data as x \
        MERGE (c:Card {{cardId:x.id}}) ON MATCH SET c.image_array = x.weights;"
    )

# ---------------------------------------------------------------------
# Add new posts to NLP document and recalculate the NLP document vector

def makedict(df):
    d = {}
    for item in df["card_id"].values:
        if item not in d.keys():
            d[item] = []
    for index, row in df.iterrows():
        cpc = row["search_cache"].replace("\\n", " ").replace("\\", " ")
        cpc = re.sub("u[1-9]....", "", cpc)
        cpc = cpc.lower()
        cpc = re.findall(r"[\w']+", cpc)
        cpc = list(filter(lambda x: x not in list(stopwords.words("english")), cpc))
        cpc = " ".join(cpc)
        d[row["card_id"]].append(cpc)
    for item in d.keys():
        d[item] = " ".join(d[item])
    return d


for stopword in list(stopwords.words("english")):
    nlp.vocab[stopword].is_stop = True

def nlp_doc_vector(df_var):
    """
    Cleans the IG post caption (search_cache) column and creates a document vector for each card_id

    Parameters
    ---
    df_var: pandas dataframe
        dataframe containing the search_cache column

    Returns
    ---
    List: A 1D array of 300 float values representing the document vector.

    """
    df_var["search_cache"] = df_var["search_cache"].str.replace('"', "")
    df_var["search_cache"] = df_var["search_cache"].str.replace('"",', "")
    df_var["search_cache"] = df_var["search_cache"].str.replace("\n", "")
    df_var["search_cache"] = df_var["search_cache"].str.replace('",', "")
    df_var.search_cache = df_var.search_cache.astype(str)
    df_var = df_var.groupby(["card_id"])["search_cache"].apply(" ".join).reset_index()
    df_var.search_cache = pd.DataFrame(df_var["search_cache"].str.replace("\n-\n", " "))
    bad_character_list = [
        "\\n•",
        "\\n",
        "\n",
        ",",
        ".",
        "?",
        "!",
        ")",
        "(",
        "#",
        "&",
        "\r",
        '"',
        "\r\n\r\n",
    ]
    for symbol in bad_character_list:
        df_var.search_cache = df_var["search_cache"].str.replace(
            symbol, "", regex=False
        )

    textdict = makedict(df_var)
    doc1 = nlp(str(textdict.values()))
    return list(doc1.vector)


nlp_df = pd.DataFrame(new_fol_card_ids_only, columns=["cardId"])
nlp_df["nlp_array"] = np.nan
nlp_df["nlp_array"] = nlp_df["nlp_array"].astype("object")
nlp_df.set_index("cardId", inplace=True)

count = 0
for card_id in new_fol_card_ids_only:
    count += 1
    posts = fetch_posts_for(card_id)
    try:
        df = pd.DataFrame(posts)
        doc1 = nlp_doc_vector(df)
        nlp_df.at[card_id, "nlp_array"] = doc1
        print(f"{count}/{len(nlp_df)} Writing array for {card_id}")
    except:
        print(f"pass on {card_id}")


# Move Card Id to a column instead of the index
nlp_df.reset_index(inplace=True)

# Remove all the cards without NaN values.
nlp_df_lite = nlp_df.loc[nlp_df["nlp_array"].notna()].copy()

# Update each node property with nlp document vector
count2 = 1
with graphDB_Driver.session() as graphDB_Session:
    for index, row in nlp_df_lite.iterrows():
        graphDB_Session.run(
            f"MATCH (c:Card {{cardId:{row['cardId']}}})\
             SET c.nlp_array = {row['nlp_array']}"
        )
        print(f"{count2}/{len(nlp_df_lite)} - Writing {index} to the neo4j database.")
        count2 += 1
        if count2 % 10000 == 0:
            graphDB_Session.sync()

# Add With_photos label to card nodes to speed up searches
with graphDB_Driver.session() as graphDB_Session:
    result = graphDB_Session.run(
        "MATCH (c:Card) WHERE reduce(total=0, number in c.image_array \
         | total + number) > 10 SET c:With_photos"
    )

# Add With_nlp label to card nodes to speed up searches
with graphDB_Driver.session() as graphDB_Session:
    result = graphDB_Session.run(
        "MATCH (c:Card) WHERE EXISTS(c.nlp_array) SET c:With_nlp"
    )

# Create Normalized array for card nodes
with graphDB_Driver.session() as graphDB_Session:
    result = graphDB_Session.run(
        f"MATCH (c:With_photos)  RETURN c.cardId AS cardId, c.image_array AS image"
    )

resultlist = [[record["cardId"], record["image"]] for record in result]
df = pd.DataFrame.from_records(resultlist, columns=["cardId", "image_array"])
df2 = pd.DataFrame(df.image_array.values.tolist(), index=df.cardId).copy()

# Min Max normalization
df2 = df2.div(df2.sum(axis=1), axis=0).copy()

count = 1
with graphDB_Driver.session() as graphDB_Session:
    for index, row in df2.iterrows():
        list_ = row.tolist()
        graphDB_Session.run(
            f"MATCH (c:Card {{cardId:{index}}})\
             SET c.norm_image_array = {list_}"
        )
        print(f"{count}/{len(df2)} - Updatating Norm Image array on {index} card node.")
        count += 1
        if count % 10000 == 0:
            graphDB_Session.sync()
    graphDB_Driver.close()
