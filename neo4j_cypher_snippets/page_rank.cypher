// Set a second label onto (:Card) for those with reviews

MATCH (u:User) WHERE SIZE( (u)-[:WROTE]->() ) > 10
SET u:Reviewer

//Load the graph into Memory
CALL algo.graph.load('review-graph',
  'MATCH (u:Reviewer) RETURN id(u) AS id',
  'MATCH (u:Reviewer)-[:WROTE]->()-[:REVIEWS]->()<-[:REVIEWS]-()<-[:WROTE]-(u2:Reviewer)
   RETURN id(u) AS source, id(u2) AS target, count(*) AS weight',
  {graph:'cypher', direction: "BOTH"});


// Compute source nodes
MATCH (u:User {id: $userId})-[:WROTE]->()-[:REVIEWS]->()<-[:REVIEWS]-()<-[:WROTE]-(other)
WITH u, other, count(*) AS count
WHERE count > 1
WITH u, collect(other) AS sourceNodes

// Execute the PageRank algorithm
CALL algo.pageRank.stream(null, null, {
  iterations:5, direction: "BOTH",
  graph: "review-graph", sourceNodes: sourceNodes
})

// Only keep users that have a PageRank score bigger than the default
YIELD nodeId, score
WITH u, algo.getNodeById(nodeId) AS node, score
WHERE score > 0.15 AND node <> u

// Keep up to 50 users
WITH u, node, score
ORDER BY score DESC
LIMIT 50

// Create a relationship between our user (u) and the influential users (node)
MERGE (u)-[trust:TRUSTS]->(node)
SET trust.score = score

//________________________________________________________________________________________


//Load the graph into Memory
CALL algo.graph.load('image-graph',
  'MATCH (u:Card) RETURN id(u) AS id',
  'MATCH (u:Card)-[:IMAGE_TOPIC]->()<-[:IMAGE_TOPIC]-(u2:Card)
   RETURN id(u) AS source, id(u2) AS target, count(*) AS weight',
  {graph:'cypher', direction: "BOTH"});


// Compute source nodes
MATCH (u:Card {cardId: $userId})-[:IMAGE_TOPIC]->()<-[:IMAGE_TOPIC]-(other)
WITH u, other, count(*) AS count
WHERE count > 1
WITH u, collect(other) AS sourceNodes

// Execute the PageRank algorithm
CALL algo.pageRank.stream(null, null, {
  iterations:5, direction: "BOTH",
  graph: "review-graph", sourceNodes: sourceNodes