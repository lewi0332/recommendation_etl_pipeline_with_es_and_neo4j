"""
Used this python script to parse words and their corresponding vectors from
the model.txt file we used for the first recommender.

Idea is to create a node for each word and add the vector as a property using
the graphaware library.

This method could still work, but we would need RAM for 3mil nodes.
"""
import io
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# Database Credentials
uri = os.environ.get(URI) # bolt+routing://protocol is used for the connection
userName = os.environ.get(USERNAME)
password = os.environ.get(PASSWORD)

# Connect to the neo4j database server
graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

def load_vectors(fname):
    """Function to parse words into a single file dict = {word:[vector], }"""

    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        # vectors = list(tokens[1:])
        data[tokens[0]] = [float(i) for i in tokens[1:]]
    return data


# ran the load vectors on the FastText library becuase it was
# better trained on standard english and smaller than my model.txt file
fastnet = load_vectors("./tmp_files/source3/wiki-news-300d-1M.txt")
test_word = fastnet["test"]


# Used this as a connection test. The Neo4j Graph needs to have a pipeline set up to tokenize words
# nlp = """
# CALL ga.nlp.processor.addPipeline({textProcessor: 'com.graphaware.nlp.processor.stanford.StanfordTextProcessor',
# name: 'customStopWords', processingSteps: {tokenize: true, ner: false, dependency: false}, stopWords: '+,result, all, during',
# threadNumber: 20});
# """
# graph.run(nlp)


# Another connection test:
# match = """MATCH (n) RETURN n LIMIT 10"""
# graph.run(match).data()


# py2neo test of .match()
# graph.match('70984')


def load_word_vectors_into_graph(fname):
    """New function that parses the word2Vec models and adds each word as
    a node with its vector as a property"""
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    # just getting pased the first line, varible not needed
    n, d = map(int, fin.readline().split())

    with graphDB_Driver.session() as graphDB_Session:

        for line in fin:
            tokens = line.rstrip().split(" ")
            graphDB_Session.run(
                f"MERGE (w:Word {{value:'{tokens[0]}'}})\
                              ON CREATE SET w.word2vec_array = {[float(i) for i in tokens[1:]]} "
            )


load_word_vectors_into_graph(
    "../recommender/recommender/tmp_files/source3/wiki-news-300d-1M.txt"
)


# FUTURE mean on all vectors:
# a = np.array([[1, 2], [3, 4], [5, 6]])
# np.mean(a, axis=1)


results = {
    "posts": [
        {
            "card_id": 2,
            "post_id": "202xxxxx",
            "photo": "https: //scontenxxxx",
            "search_cache": "@us_themmusic 🔥 the delancey repellendus-placeat my secret life 126 elizabeth collier ",
            "published_at": "2019-04-17T21: 36: 42.000-04: 00",
            "provider": "instagram",
            "interactions": 10,
            "impressions": 24086,
            "permalink": "https: //www.instagram.com/p/xxx/",
            "mentions": [],
            "hashtags": "",
            "type": "image",
            "video": "",
            "unique_stats": {"likes_count": 10, "comments_count": 0},
            "caption": "@xxxxx 🔥",
            "username": "xxx",
            "id": 20106,
            "categories": [30],
        },
        {
            "card_id": 3,
            "post_id": "2xxxxx",
            "photo": "https:xxxxx",
            "search_cache": "favorite color of the season 🍊🙃 los angeles, california ",
            "published_at": "2019-08-02T10: 47: 03.000-04: 00",
            "provider": "instagram",
            "interactions": 99,
            "impressions": 24086,
            "permalink": "https: //www.instagram.com/p/xxxx/",
            "mentions": "xxxx",
            "hashtags": "",
            "type": "image",
            "video": "",
            "unique_stats": {"likes_count": 91, "comments_count": 8},
            "caption": "Favorite color of the season 🍊🙃",
            "username": "xxx",
            "id": 20640,
            "categories": [10, 38, 33],
        },
    ]
}

with graphDB_Driver.session() as graphDB_Session:

    # Create nodes
    for line in results["posts"]:
        # py2Neo obect with everything it needs to build a node
        graphDB_Session.run(
            f"MERGE (c:Card {{cardId:{line['card_id']}}}) ON CREATE SET c.userName = '{line['username']}'"
        )
        graphDB_Session.run(
            f"WITH {line['categories']} as cats  MATCH (c:Card), (cat:Category) WHERE c.cardId = {line['card_id']} AND cat.cat in cats \
        MERGE (c)-[:CONTENT_TOPIC]->(cat)"
        )
