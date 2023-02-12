import os
import nltk
import re
from nltk.corpus import stopwords
import spacy
from elasticsearch import Elasticsearch
import pandas as pd
import time
from neo4j import GraphDatabase
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Database Credentials
uri = os.environ.get(URI) # bolt+routing://protocol is used for the connection
userName = os.environ.get(USERNAME)
password = os.environ.get(PASSWORD)

es_uri = os.environ.get(ES_URI)

# Connect to the neo4j database server
graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

nltk.download("stopwords")

"""Move the 11gb file model.txt to the CWD and run this in terminal"""
# python -m spacy init-model en /tmp/custom_vectors --vectors-loc model.txt

"""Loading dictionary of words into Spacy"""
nlp = spacy.load("/tmp/custom_vectors")


"""Import new posts here"""
# get all the cardIds on the database:

with graphDB_Driver.session() as graphDB_Session:
    all_cardIds = graphDB_Session.run("MATCH (c:Card) RETURN c.cardId").values()


def fetch_posts_for(card_id):
    """INPUT: card ID as int Elastic search query to
       return instagram post data for the last 33 posts"""
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
    # print(instagram_only_clause)
    body = {"sort": [{"published_at": "desc"}], "query": instagram_only_clause}
    result = es.search(index="posts", size=33, body=body)
    hits = result["hits"]["hits"]
    return [x["_source"] for x in hits]


es = Elasticsearch([es_uri])


def makedict(df):
    """
    Creates a dictionary of card_id keys and a concatenated string of
    the all the text from their last 33 posts as a value. Removing 
    stop words and other non-words characters.

    Parameters
    ---
    df: pandas dataframe
        dataframe of card_id and search_cache

    Returns
    ---
    d: dictionary
        dictionary of card_id keys and a concatenated string of text
    """

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


def maketokens(textdict):
    """
    Creates a dictionary of card_id keys and a spacy tokenized string of
    the all the text from their last 33 posts as a value.

    Parameters
    ---
    textdict: dictionary
        dictionary of card_id keys and a concatenated string of text
    
    Returns
    ---
    x: dictionary
        dictionary of card_id keys and a spacy tokenized string of text
    """
    xl = [value for value in textdict.values()]
    xk = [key for key in textdict.keys()]
    x = {}
    for index, text in enumerate(xl):
        x[xk[index]] = nlp(text)
    return x


# Remove stop words from spacy model
for stopword in list(stopwords.words("english")):
    nlp.vocab[stopword].is_stop = True


def nlp_sim(df_var):
    """
    Creates a document vector for each card_id. 

    Parameters
    ---
    df_var: pandas dataframe
        dataframe of card_id and post text

    Returns
    ---
    document vector: list
        list of document vectors
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
        df_var.search_cache = pd.DataFrame(
            df_var["search_cache"].str.replace(symbol, "", regex=False)
        )
    textdict = makedict(df_var)
    doc1 = nlp(str(textdict.values()))
    return list(doc1.vector)

# ---------------------------------------------------------------------
# create an empty dataframe of cardIds

nlp_df = pd.DataFrame(all_cardIds, columns=["cardId"])
nlp_df["nlp_array"] = np.nan
nlp_df["nlp_array"] = nlp_df["nlp_array"].astype("object")
nlp_df.set_index("cardId", inplace=True)

# ---------------------------------------------------------------------
# fetch posts for each cardId and create a document vector for each cardId

start = time.time()
count = 1
for card_id in all_cardIds:
    card_id = card_id[0]
    print(f"{count}/{len(nlp_df)} Writing array for {card_id}")
    count += 1
    posts = fetch_posts_for(card_id)
    # print(f"Fetching post: {time.time()-start:.4f}secs")
    # next = time.time()
    try:
        df = pd.DataFrame(posts)
        # print(f"\nCreating DF took: {time.time()-next:.4f}secs")
        # next2=time.time()
        doc1 = nlp_sim(df)
        # print(f"\nnlp_sim function took: {time.time()-next2:.4f}secs")
        # next3=time.time()
        nlp_df.at[card_id, "nlp_array"] = doc1
        # print(f"\nAdding the array took: {time.time()-next3:.4f}secs")
        print(f"{count}/{len(nlp_df)} Writing array for {card_id}")
    except:
        print(f"pass on {card_id}")


# Move Card Id to a column instead of the index
nlp_df.reset_index(inplace=True)
# Remove all the cards with NaN values.
nlp_df_lite = nlp_df.loc[nlp_df["nlp_array"].notna()].copy()

# ---------------------------------------------------------------------
# Update each node property with nlp document vector
count2 = 1
with graphDB_Driver.session() as graphDB_Session:
    for index, row in nlp_df_lite.iterrows():
        graphDB_Session.run(
            f"MATCH (c:Card {{cardId:{row['cardId']}}})\
             SET c.nlp_array = {row['nlp_array']}"
        )
        print(f"{count2}/{len(nlp_df)} - Writing {index} to the database.")
        count2 += 1
        # sync session every 5000 iterations 
        if count2 % 5000 == 0:
            graphDB_Session.sync()
graphDB_Driver.close()


# ---------------------------------------------------------------------
# Potential refactor
"""
- Import current dict of words from neo4j
- Add the new bag of words to the word node
- Calculate the document vector and add to (:Card)
- Create df with CardID and 300 point vector
- Separate graph?"""

# Json import
"""
USE JSON to import this massive pile
CALL apoc.periodic.iterate("
WITH '' as URL
CALL apoc.load.json(URL) YIELD value return value",

"MATCH (c:Card {cardId:value.cardId}), (img:Imagetype {cat:value.cat}) 
MERGE (c)-[r:IMAGE_TOPIC]->(img) 
ON CREATE SET r.count = 1
ON MATCH SET r.count = r.count + 1", {batchSize:10000});"""
