import json
import numpy as np
import os
import requests
from neo4j import GraphDatabase
from time import sleep
from dotenv import load_dotenv
import os

load_dotenv()

# Database Credentials
uri = os.environ.get(URI) # bolt+routing://protocol is used for the connection
userName = os.environ.get(USERNAME)
password = os.environ.get(PASSWORD)

with open('big_card.json') as json_file:
    results = json.load(json_file)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Connect to the neo4j database server
graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

with graphDB_Driver.session() as graphDB_Session:
    graphDB_Session.run("""
        LOAD CSV WITH HEADERS
        FROM 'uri to simple influencer topic csv file on github'
        AS line
        MERGE(c: Category {cat: line.Cat})
        ON CREATE SET c.description=line.Desc;
    """)

    graphDB_Session.run("""// photo category create
        LOAD CSV WITH HEADERS
        FROM 'uri to simple photo cat list csv file on github'
        AS line
        MERGE(c: Imagetype {cat: toInt(line.cat)})
        ON CREATE SET c.description=line.description
    """)
    graphDB_Session.run(
        """CREATE CONSTRAINT ON (n:Card) ASSERT n.cardId IS UNIQUE""")

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
