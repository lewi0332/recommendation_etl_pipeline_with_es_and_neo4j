"""
This is a devolpment script to test the neo4j graph database and the neo4j python driver 
with the built-in Euclidean and Cosine similarity algorithms in Neo4j.

Databased is hosted on AWS using the bolt+routing protocol with a 3 node cluster.

This script is not used in the production code.

Author: Derrick Lewis 
Date: 2019-08-01
"""
import os
import pandas as pd
import time
from neo4j import GraphDatabase

start = time.time()

uri =os.environ.get("AWS_BOLT")
userName = os.environ.get("AWS_USER")
password = os.environ.get("AWS_KEY")
graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

 # test a target cardId
target = [42508]

# get the target cardId follower count to set a range for the query
with graphDB_Driver.session() as graphDB_Session:
    target_id_followers = graphDB_Session.run(
        f"MATCH (p:With_photos {{cardId:{target}}}) RETURN p.igFollower"
    ).values()

min_followers = target_id_followers[0][0] * 0.7
max_followers = target_id_followers[0][0] * 1.3

print(f"Start graph driver: {time.time()-start:.3f}sec")
next = time.time()
with graphDB_Driver.session() as graphDB_Session:
    image = graphDB_Session.run(f"MATCH (p:With_photos) WHERE p.igFollower > {min_followers} AND p.igFollower < {max_followers} \
            WITH {{item:id(p), card: p.cardId, weights:p.norm_image_array}} as userData \
            WITH collect(userData) as personIMAGE WITH personIMAGE, [value in personIMAGE \
            WHERE value.card IN {target} | value.item ] AS sourceIds \
            CALL algo.similarity.cosine.stream(personIMAGE, {{sourceIds: sourceIds, topK:1000}}) \
            YIELD item1, item2, similarity \
            with algo.getNodeById(item1) AS from, algo.getNodeById(item2) as to, similarity \
            RETURN from.cardId as from, to.cardId as to, similarity \
            ORDER BY similarity DESC"
    ).values()
    print(f"\n\nImage Cosine took this long: {time.time() - next:.3f}sec")
    next2 = time.time()

    nlp = graphDB_Session.run(
        f"MATCH (p:With_nlp) WHERE p.igFollower > {min_followers} AND p.igFollower < {max_followers}  \
            WITH {{item:id(p), card: p.cardId, weights:p.nlp_array}} as userData \
            WITH collect(userData) as personNLP WITH personNLP, [value in personNLP \
            WHERE value.card IN {target} | value.item ] AS sourceIds \
            CALL algo.similarity.cosine.stream(personNLP, {{sourceIds: sourceIds, topK:1000}}) \
            YIELD item1, item2, similarity \
            with algo.getNodeById(item1) AS from, algo.getNodeById(item2) as to, similarity \
            RETURN from.cardId as from, to.cardId as to, similarity \
            ORDER BY similarity DESC"
    ).values()
    # nlp = graphDB_Session.run(
    #     f"MATCH (from:With_nlp)-[sim:NLP_SIMILARITY]-(to) \
    #         WHERE from.igFollower > {min_followers} AND from.igFollower < {max_followers} AND from.cardId IN {target} \
    #         RETURN from.cardId as from, to.cardId as to, sim.score as similarity \
    #         ORDER BY similarity DESC"
    # ).values()
    print(f"\n\nText Cosine took this long: {time.time()-next2:.3f}sec")
    next3 = time.time()

df_image = pd.DataFrame(image, columns=["from", "to", "similarity"])
df_nlp = pd.DataFrame(nlp, columns=["from", "to", "similarity"])
print(f"\n\nCreate DataFrames: {time.time()-next3:.3f}sec")
next4 = time.time()

new_combined = pd.merge(df_image, df_nlp,  how='left', left_on=['to','from'], right_on = ['to','from'])
new_combined.fillna(0, inplace=True)
new_combined["total"] = new_combined["similarity_x"] + new_combined["similarity_y"]
new_combined.sort_values(by=['from', 'total'], inplace=True, ascending=False)

print(f"\n\nJoin DataFrames: {time.time()-next4:.3f}sec")
next5 = time.time()

final = new_combined.groupby('from').head(20)

print(final)
# final.to_csv('first_trial_600.csv', index=False)

print(f"The rest of this took {time.time()-next5:.3f}sec")
print(f"Total time: {time.time()-start:3f}sec")


# ---------------------------------------------------------------------
# Below is another method to normalize the similarity scores
# Better to skip importing another package and just use the above method

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# x = df_nlp[["similarity"]].values.astype(float)
# df_nlp["similarity"] = scaler.fit_transform(x)
# df_nlp["similarity"] = df_nlp["similarity"].div(-1).add(1)

# scaler = MinMaxScaler()
# x = df_image[["similarity"]].values.astype(float)
# df_image["similarity"] = scaler.fit_transform(x)
# df_image["similarity"] = df_image["similarity"].div(-1).add(1)
