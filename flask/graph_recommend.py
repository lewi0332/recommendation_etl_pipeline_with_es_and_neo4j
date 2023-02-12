from neo4j import GraphDatabase
import pandas as pd
import time
import os


def get_recommendations(target, min_followers=None, max_followers=None):
    """
    Returns a list of recommended card_ids

    Parameters
    ---
    card_id: int
        The card_id of the user to get recommendations for
    min: int
        The minimum number of followers the user should have
    max: int
        The maximum number of followers the user should have

    Returns
    ---
    dict: 
        A list of recommended card_ids and their similarity scores
    """

    uri = os.environ.get("AWS_BOLT")
    userName = os.environ.get("AWS_USER")
    password = os.environ.get("AWS_KEY")
    graphDB_Driver = GraphDatabase.driver(uri, auth=(userName, password))

    with graphDB_Driver.session() as graphDB_Session:
        if min_followers is None:
            target_id_followers = graphDB_Session.run(
                f"MATCH (p:With_photos {{cardId:{target}}}) \
                RETURN p.igFollower").values()
            min_followers = target_id_followers[0][0] * 0.7
        else:
            pass
        if max_followers is None:
            max_followers = target_id_followers[0][0] * 1.3
        else:
            pass
        image = graphDB_Session.run(f"MATCH (p:With_photos) \
            WHERE p.igFollower > {min_followers} AND \
            p.igFollower < {max_followers} \
            WITH {{item:id(p), card: p.cardId, weights:p.norm_image_array}} \
            as userData \
            WITH collect(userData) as personIMAGE \
            WITH personIMAGE, [value in personIMAGE \
            WHERE value.card IN {[target]} | value.item ] \
            AS sourceIds CALL algo.similarity.cosine.stream(personIMAGE, \
            {{sourceIds: sourceIds, topK:1000}}) \
            YIELD item1, item2, similarity \
            WITH algo.getNodeById(item1) AS from, \
            algo.getNodeById(item2) as to, similarity \
            RETURN from.cardId as from, to.cardId as to, similarity \
            ORDER BY similarity DESC").values()

        nlp = graphDB_Session.run(f"MATCH (p:With_nlp) \
            WHERE p.igFollower > {min_followers} AND \
            p.igFollower < {max_followers}  \
            WITH {{item:id(p), card: p.cardId, weights:p.nlp_array}} \
            as userData \
            WITH collect(userData) as personNLP WITH personNLP, \
            [value in personNLP \
            WHERE value.card IN {[target]} | value.item ] AS sourceIds \
            CALL algo.similarity.cosine.stream(personNLP, \
            {{sourceIds: sourceIds, topK:1000}}) \
            YIELD item1, item2, similarity \
            with algo.getNodeById(item1) AS from, \
            algo.getNodeById(item2) as to, similarity \
            RETURN from.cardId as from, \
                to.cardId as to, similarity \
            ORDER BY similarity DESC").values()

        '''
        The following query uses pre-computed nlp similarity numbers
        to speed things up.Substitute this for the query above if the
        NLP_SIMILARITY relationships are reliable in the neo4j database
        '''
        # nlp = graphDB_Session.run(
        #   f"MATCH (from:With_nlp)-[sim:NLP_SIMILARITY]-(to) \
        #   WHERE from.igFollower > {min_followers} AND from.igFollower \
        #   < {max_followers} AND from.cardId IN {target} \
        #   RETURN from.cardId as from, to.cardId as to, \
        #        sim.score as similarity \
        #   ORDER BY similarity DESC").values()

    df_image = pd.DataFrame(image, columns=["from", "to", "similarity"])
    df_nlp = pd.DataFrame(nlp, columns=["from", "to", "similarity"])

    new_combined = pd.merge(df_image, df_nlp,  how='left',
                            left_on=['to', 'from'], right_on=['to', 'from'])
    new_combined.fillna(0, inplace=True)
    new_combined.columns = ['from', 'to', 'image_similarity', 'nlp_similarity']
    new_combined["total"] = (new_combined["image_similarity"] +
                             new_combined["nlp_similarity"])

    new_combined.sort_values(by=['from', 'total'], inplace=True,
                             ascending=False)

    return new_combined.head(20).to_dict('index')
