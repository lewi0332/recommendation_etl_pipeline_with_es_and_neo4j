import time
from elasticsearch import Elasticsearch

def es_iterate_all_documents(es, index, pagesize=250, scroll_timeout="1m", **kwargs):
    """
    Helper to iterate ALL values from a single index
    Yields all the documents.
    """
    is_first = True
    instagram_only_clause = { "filtered": { "filter": { "bool": { "must": [ { "term": { "provider": "instagram" } } ] } } } }
    while True:
        # Scroll next
        if is_first: # Initialize scroll
            body = {
                "size": pagesize,
                "query": instagram_only_clause
            }
            result = es.search(index=index, scroll="1m", **kwargs, body={
            })
            is_first = False
        else:
            body = {
                "scroll_id": scroll_id,
                "scroll": scroll_timeout
            }
            result = es.scroll(body=body)
        scroll_id = result["_scroll_id"]
        hits = result["hits"]["hits"]
        # Stop after no more docs
        if not hits:
            break
        # Yield each entry
        yield from (hit['_source'] for hit in hits)

        time.sleep(5)

es = Elasticsearch(["elastic search uri"])

for entry in es_iterate_all_documents(es, 'posts'):
    print(entry)
