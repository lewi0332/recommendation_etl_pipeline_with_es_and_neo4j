WITH "
MATCH (c:Card)-[r:IMAGE_TOPIC]->(img:Imagetype)
WHERE c.cardId < 5000
RETURN id(c) AS item, id(img) AS category, r.count AS weight" AS query

CALL algo.similarity.cosine(
query, {
  graph: "cypher",
  topK:1000,
  similarityCutoff: 0.5,
  write: true ,
  writeRelationshipType:'IMAGE_SIM',
  writeProperty:'score',
  writeBatchSize:1
}
)
YIELD nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, stdDev, p25, p50, p75, p90, p95, p99, p999, p100
RETURN nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, stdDev, p25, p50, p75, p90, p95, p99, p999, p100
;

// 4794 @ 2904ms
// 9382 @ 5999ms
// 18924 @ 21881ms
// 34527 @ 72982ms

CALL apoc.periodic.iterate(
'
WITH "
MATCH (c:Card)-[r:IMAGE_TOPIC]->(img:Imagetype)
RETURN id(c) AS item, id(img) AS category, r.count AS weight"
 AS query
RETURN query',

'CALL algo.similarity.cosine(query, {graph: "cypher", topK:1000, similarityCutoff: 0.5,
write: true , writeRelationshipType:"IMAGE_SIM", writeProperty:"score", writeBatchSize:1})
YIELD nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean,
stdDev, p25, p50, p75, p90, p95, p99, p999, p100
RETURN nodes, similarityPairs, write,
writeRelationshipType, writeProperty, min, max, mean, stdDev, p25, p50, p75, p90, p95, p99, p999, p100', { batchSize:100 })

CALL apoc.periodic.iterate('MATCH(p1:Card) MATCH(p2:Card) WHERE p2 <> p1  RETURN p1, p2',
'MERGE (p1)-[r:IMAGE_SIMILARITY]->(p2) ON CREATE SET r.score = algo.similarity.cosine(p1.image_array, p2.image_array)', { batchSize:100 })
