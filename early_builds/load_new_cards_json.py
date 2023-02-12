"""
This script tests the loaging from a json file into the neo4j database.
"""
import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Database Credentials

uri = os.environ.get(URI) # bolt+routing://protocol is used for the connection
userName = os.environ.get(USERNAME)
password = os.environ.get(PASSWORD)

with open("big_card.json") as json_file:
    results = json.load(json_file)

# Create catgegory nodes and Imagetype nodes
graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

with graphDB_Driver.session() as graphDB_Session:
    graphDB_Session.run(
        """LOAD CSV WITH HEADERS
        FROM 'uri to simple influencer topic csv file on github'
        AS line
        MERGE(c: Category {cat: line.Cat})
        ON CREATE SET c.description=line.Desc;
    """
    )

    graphDB_Session.run(
        """// photo category create
        LOAD CSV WITH HEADERS
        FROM 'uri to simple photo cat list csv file on github'
        AS line
        MERGE(c: Imagetype {cat: toInt(line.cat)})
        ON CREATE SET c.description=line.description
    """
    )
    graphDB_Session.run("""CREATE CONSTRAINT ON (n:Card) ASSERT n.cardId IS UNIQUE""")


count = 1
for lines in results:
    with graphDB_Driver.session() as graphDB_Session:
        for line in lines:
            print(line["id"])
            graphDB_Session.run(f"MERGE (c:Card {{cardId:{line['id']}}})")
            count += 1
            if count % 5000 == 0:
                graphDB_Session.sync()
    graphDB_Session.close()

count2 = 1
for lines in results, 10000:
    with graphDB_Driver.session() as graphDB_Session:
        for line in lines:
            graphDB_Session.run(
                f"WITH {line['tags']} as cats  MATCH (c:Card), (cat:Category) \
                                      WHERE c.cardId = {line['id']} AND cat.cat in cats \
                                      MERGE (c)-[:CONTENT_TOPIC]->(cat)"
            )
            count2 += 1
            if count2 % 5000 == 0:
                graphDB_Session.sync()
    graphDB_Session.close()


with graphDB_Driver.session() as graphDB_Session:
    graphDB_Session.run(
        "MATCH (cat:Category) WITH cat ORDER BY cat.cat WITH collect(cat) AS cats \
            MATCH (p:Card) \
            SET p.topic_array = algo.ml.oneHotEncoding(cats, [(p)-[:CONTENT_TOPIC]->(cat) | cat])"
    )
