"""
This script is not used in the production code.

This is a one-time-use script to scale all values in the image array of all cards 
to a normalized array of values between 0 and 1.
"""
import os
import pandas as pd
from neo4j import GraphDatabase

uri = os.environ.get(GRAPHENE_BOLT)
userName = os.environ.get(GRAPHENE_USER)
password = os.environ.get(GRAPHENE_KEY)

graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

with graphDB_Driver.session() as graphDB_Session:
    result = graphDB_Session.run(
        "MATCH (c:With_photos) RETURN c.cardId AS cardId, c.image_array AS image"
    )

resultlist = [[record["cardId"], record["image"]] for record in result]
df = pd.DataFrame.from_records(resultlist, columns=["cardId", "image_array"])
df2 = pd.DataFrame(df.image_array.values.tolist(), index=df.cardId).copy()

df2 = df2.div(df2.sum(axis=1), axis=0).copy()

count = 1
with graphDB_Driver.session() as graphDB_Session:
    for index, row in df2.iterrows():
        list = [
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
            row[10],
            row[11],
            row[12],
            row[13],
            row[14],
            row[15],
            row[16],
            row[17],
            row[18],
            row[19],
            row[20],
            row[21],
            row[22],
            row[23],
            row[24],
            row[25],
            row[26],
            row[27],
            row[28],
            row[29],
            row[30],
            row[31],
            row[32],
        ]
        graphDB_Session.run(
            f"MATCH (c:Card {{cardId:{index}}})\
             SET c.norm_image_array = {list}"
        )
        print(f"{count}/{len(df2)} - Writing {index} to the database.")
        count += 1
        if count % 5000 == 0:
            graphDB_Session.sync()
