import os
from neo4j import GraphDatabase
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn import preprocessing
from dotenv import load_dotenv

load_dotenv()

# Database Credentials
uri = os.environ.get(URI) # bolt+routing://protocol is used for the connection
userName = os.environ.get(USERNAME)
password = os.environ.get(PASSWORD)

graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

# return cardId's and image arrays
with graphDB_Driver.session() as graphDB_Session:
    results = graphDB_Session.run(
        f"MATCH (c:Card) RETURN c.cardId, c.image_array")

df = pd.DataFrame([r for r in results], columns=['cardId', 'images'])

# convert image arrays into columns
df2 = pd.DataFrame(df.images.values.tolist(), index=df.cardId)

df2.iloc[6:7, :].sum(axis=1)

# normalize each row by total amount of images
scaler = preprocessing.normalize(df2, norm='l1', axis=1)
scaled_df = pd.DataFrame(
    scaler, columns=df2.columns, index=df2.index)

# calculate similarity
df_image_similarity = pd.DataFrame(cosine_similarity(scaled_df),
                                   index=df2.index, columns=df2.index)

# write Similarity relationships to Neo4j

df_image_similarity.head()

with graphDB_Driver.session() as graphDB_Session:
    for index, row in df_image_similarity.iterrows():
        print('''
          MATCH (a:Card {cardId:$label1})), (b:Card {cardId:$label1})
          MERGE (a)-[r:TOPIC_SIM {score:$label3}]->(b)
        ''', parameters={'label1': index, 'label2': row, 'label3': row.index})
