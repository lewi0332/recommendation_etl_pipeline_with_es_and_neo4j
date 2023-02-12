''' This is my scratchpad for all cypher queries to build the test graph'''


//training on Cypher
:play intro-neo4j-exercises

//Install plugins: APOC, Graph Algorithms, Graphaware Framework, NLP x3 


// To use Graphaware NLP libraries add these to the plugins folder:
//      neo4j-framework (the JAR for this is labeled "graphaware-server-enterprise-all")
//      neo4j-nlp
//      neo4j-nlp-stanfordnlp
//      https://stanfordnlp.github.io/CoreNLP/#download

// Then add the following to bottem of the config file: 

//      dbms.unmanaged_extension_classes=com.graphaware.server=/graphaware
//      com.graphaware.runtime.enabled=true
//      com.graphaware.module.NLP.1=com.graphaware.nlp.module.NLPBootstrapper
//      dbms.security.procedures.whitelist=ga.nlp.*, apoc.*, algo.*,


CREATE CONSTRAINT ON (n:Card) ASSERT n.cardId IS UNIQUE;
CREATE CONSTRAINT ON (img:Imagetype) ASSERT img.cat IS UNIQUE;


LOAD CSV WITH HEADERS 
FROM 'file:///post_df.csv'
AS line
MERGE (cat:PhotoCat {style:line.photo_cat});

LOAD CSV WITH HEADERS 
FROM 'file:///post_df.csv'
AS line
MERGE (p:Person {personId:line.user});

LOAD CSV WITH HEADERS 
FROM 'file:///post_df.csv'
AS line
MATCH (cat:PhotoCat {style:line.photo_cat}), (p:Person {personId:line.user})
MERGE  (p)-[rel:POSTED_ABOUT]->(cat)
ON CREATE SET rel.count = 1
ON MATCH SET rel.count =rel.count + 1;

LOAD CSV WITH HEADERS 
FROM 'file:///post_df.csv'
AS line
MERGE (m:Post {postId:line.post_id})ON CREATE SET m.postMentions = line.post_mentions, 
m.photoCat=line.photo_cat, 
m.postHashtags = line.post_hashtags, 
m.imageLink = line.image_link, 
m.categories=line.categories, 
m.description=line.description, 
m.post_followers=line.post_followers, 
m.postMentions=line.post_mentions, 
m.postCaption=line.post_caption, 
m.allText=line.all_text;

LOAD CSV WITH HEADERS 
FROM 'file:///post_df.csv'
AS line
MATCH (p:Person {personId:line.user}), (m:Post {postId:line.post_id})
MERGE (p)-[:POSTED {date:datetime('1979-02-23T12:05:35.556+0100')}]->(m)



//Euc distance from single user

MATCH (p:Card), (img:ImageType)
OPTIONAL MATCH (p)-[posted:IMAGE_TOPIC]->(img)
WITH {item:id(p), personId: p.personId, weights: collect(coalesce(posted.count, algo.NaN()))} as userData
WITH collect(userData) as personCategory
WITH personCategory,
     [value in personCategory WHERE value.cardId IN ["37638"] | value.item ] AS sourceIds
CALL algo.similarity.euclidean.stream(personCategory, {sourceIds: sourceIds, topK: 6})
YIELD item1, item2, similarity
WITH algo.getNodeById(item1) AS from, algo.getNodeById(item2) AS to, similarity
RETURN from.personId AS from, to.personId AS to, similarity
ORDER BY similarity DESC

// Jaccard Similarity from a single user -- Only counts the Category once!!! 

MATCH (p1:Person {personId: '37638'})-[:POSTED_ABOUT]->(photoCat)
WITH p1, collect(id(photoCat)) AS p1cat
MATCH (p2:Person)-[:POSTED_ABOUT]->(photoCat2) WHERE p1 <> p2
WITH p1, p1cat, p2, collect(id(photoCat2)) AS p2cat
RETURN p1.personId AS from,
       p2.personId AS to,
       algo.similarity.jaccard(p1cat, p2cat) AS similarity
ORDER BY similarity DESC




''' NLP side of the reccomendation ''' 

// Early test with pre cleaned text file. 
// //load text
// LOAD CSV WITH HEADERS
// FROM 'file:///200_text.csv' AS line
// MERGE (t:Text { text: line.all_text })
// MERGE (p:Person { personId: line.id })
// MERGE (p)-[:WROTE]->(t);

CREATE CONSTRAINT ON (n:AnnotatedText) ASSERT n.id IS UNIQUE;
CREATE CONSTRAINT ON (n:Tag) ASSERT n.id IS UNIQUE;
CREATE CONSTRAINT ON (n:Sentence) ASSERT n.id IS UNIQUE;
CREATE INDEX ON :Tag(value);

// Create an nlp pipeline to annotate text it finds in posts. Set paramaters here, add a NLP processer, add custom stopwords etc. 
CALL ga.nlp.processor.addPipeline({textProcessor: 'com.graphaware.nlp.processor.stanford.StanfordTextProcessor', 
name: 'customStopWords', processingSteps: {tokenize: true, ner: false, dependency: false}, stopWords: '+,result, all, during', 
threadNumber: 20});

// Set the pipeline we previously create to be the default we use. 
CALL ga.nlp.processor.pipeline.default("customStopWords");

CALL ga.nlp.config.setDefaultLanguage('en');

// Catch language check errors. 
CALL ga.nlp.config.set('SETTING_fallbackLanguage', 'en')

MATCH (a:AnnotatedText) with collect(a) as list
CALL ga.nlp.ml.similarity.cosine({input:list})
YIELD result RETURN result

CALL ga.nlp.ml.word2vec.load(<maxNeighbors>, <modelName>)

// Use this for the traditional method of annotating and tokenizing text. 

// MATCH (n:News)
// CALL ga.nlp.annotate({text: n.text, id: id(n), checkLanguage: false})
// YIELD result
// MERGE (n)-[:HAS_ANNOTATED_TEXT]->(result)
// RETURN result

//But! use this one for now as it iterates on our big set of data to annotate. APOC :
CALL apoc.periodic.iterate(
"MATCH (p:Post) RETURN p",
"CALL ga.nlp.annotate({text: p.allText, id: id(p), checkLanguage: false})
YIELD result MERGE (p)-[:HAS_ANNOTATED_TEXT]->(result)", {batchSize:1, iterateList:true})

// Word2Vec stuff below
CALL ga.nlp.ml.word2vec.addModel("/tmp_files/source2", "/tmp_files/index", "numberbatch");

CALL ga.nlp.ml.word2vec.listModels;

// Load the wrod2vec file into memory trial
CALL ga.nlp.ml.word2vec.load(<maxNeighbors>, <modelName>)

// Test word2vec model on two words
WITH ga.nlp.ml.word2vec.wordVector('love', 'neue_model') AS love,
ga.nlp.ml.word2vec.wordVector('hate', 'neue_model') AS hate
RETURN ga.nlp.ml.similarity.cosine(love, hate) AS similarity;

//Can't get this to work. If so, it might help me add word2Vec Vectors onto the tagged nodes from  the annotation task above.
CALL apoc.periodic.iterate(
"MATCH (t:Tag) RETURN t",
"CALL ga.nlp.ml.word2vec.attach({query: 'MATCH (t:Tag) RETURN t', modelName:'neue_model'})
YIELD result MERGE (n {word2Vec:result})", {batchSize:1, iterateList:true})


//find all the Graphaware commands
CALL dbms.procedures() YIELD name, signature, description
WHERE name =~ 'ga.nlp.*'
RETURN name, signature, description ORDER BY name asc;

CALL ga.nlp.ml.word2vec.nn('king', 10, 'neue_model') YIELD word, distance RETURN word, distance

// Create cosine similarity betweeen Annotated text nodes
MATCH (a:AnnotatedText)
with collect(a) as nodes
CALL ga.nlp.ml.similarity.cosine({input: nodes, 
query: null, relationshipType: "CUSTOM_SIMILARITY"}) YIELD result
RETURN result

//Possible the way to specifically use word2Vec in cosine sim: 

// MATCH (a:Tag:VectorContainer)
// WITH collect(a) as nodes
// CALL ga.nlp.ml.similarity.cosine({
// input:nodes, 
// property:'word2vec'})
// YIELD result
// return result;

//Doc2vec in neo4j with graphaware

CALL ga.nlp.vector.train({type:'doc2Vec', parameters:{query: "MATCH (n:Train) 
WHERE length(n.text) > 10 WITH n, rand() as r ORDER BY r 
RETURN n.text as text, id(n) as docId", iterations: 15, epochs: 10, layerSize: 400}}) YIELD result
return result

CALL apoc.periodic.iterate(
'MATCH (s:Wikipage) WHERE length(s.text) > 10 return s',
"CALL ga.nlp.vector.compute({node: s,
type:'doc2Vec',
property:'doc2vec_dim400',
label:'Doc2VecContainerSentence',
parameters: {query:'MATCH (n:HandPicked) WHERE id(n) = {id} RETURN n.text as text', iterations: 10, useExistingAnnotation: False}})
YIELD result
return result", {batchSize: 10, parallel: false})

//alternative:
CALL ga.nlp.vector.train({type:'doc2Vec', parameters:{query: "match (n:Post)-[:HAS_ANNOTATED_TEXT]->
(a:AnnotatedText)-[:CONTAINS_SENTENCE]->(:Sentence)-[:SENTENCE_TAG_OCCURRENCE]->(to:TagOccurrence)-
[:TAG_OCCURRENCE_TAG]->(t:Doc2VecVocabulary) with n, t, to order by to.startPosition asc return id(n) 
as docId, collect(t.value) as tokens", iterations: 15, epochs: 5, layerSize: 400, useExistingAnnotation: True}}) YIELD result
return result

match (:Post)
with count(*) as total_docs 
match (:Train)-[:HAS_ANNOTATED_TEXT]->(a:AnnotatedText)-[:CONTAINS_SENTENCE]->(:Sentence)-[r:HAS_TAG]->(t:Tag)
with t, count(distinct a) as n_docs, sum(r.tf) as sum_tf, total_docs
where n_docs > 1 and n_docs * 1.0/total_docs < 0.4
with t
order by n_docs desc, sum_tf desc
limit 700000
set t:Doc2VecVocabulary



/////////// CATEGORY Sim

MATCH (p1:Card {userName: 'Michael'})-[posts2:CONTENT_TOPIC]->(category)
MATCH (p2:Card)-[posts2:CONTENT_TOPIC]->(category) WHERE p2 <> p1
RETURN p1.userName AS from,
       p2.userName AS to,
       algo.similarity.cosine(collect(posts2.score), collect(likes2.score)) AS similarity
ORDER BY similarity DESC



// one hot encoding topics
MATCH (cat:Category)
WITH cat ORDER BY cat.cat
WITH collect(cat) AS cats
MATCH (p:Card)
SET p.topic_array = algo.ml.oneHotEncoding(cats, [(p)-[:CONTENT_TOPIC]->(cat) | cat]);

//write sim from all users?
CALL apoc.periodic.iterate("MATCH (c:Card) WITH {item:id(c), weights: c.topic_array} as userData WITH collect(userData) as data RETURN data",
"CALL algo.similarity.cosine(data, {topK:200, similarityCutoff: 0.5, write:true, writeRelationshipType:'topicSim', writeProperty:'score'}) YIELD  write RETURN write", {batchSize:10000})
// or? 
CALL apoc.periodic.iterate('MATCH (c:Cuisine) WITH {item:id(c), weights: c.embedding} as userData WITH collect(userData) as data RETURN data', 
'CALL algo.similarity.cosine.stream(data, {skipValue: null}) YIELD item1, item2, count1, count2, similarity WITH  algo.asNode(item1) as to, algo.asNode(item2) as from, similarity MERGE (to)-[r:SIM]->(from) ON CREATE SET r.score = similarity', {batchSize:1})

// WHat about totalling the array and removing anyone without any images: 
MATCH (c:Card)
WHERE reduce(total=0, number in c.image_array | total + number) > 20



//use Graph Projection to prewrite sim to all users? 
WITH "MATCH (c:Card)-[r:IMAGE_TOPIC]->(img:Imagetype)
      RETURN id(c) AS item, id(img) AS category, r.count AS weight" AS query
CALL algo.similarity.cosine(query, {graph: "cypher", topK:1000, similarityCutoff: 0.5, 
write:true, writeRelationshipType:'topicSim', writeProperty:'score', writeBatchSize:100}) YIELD nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, stdDev, p25, p50, p75, p90, p95, p99, p999, p100 RETURN nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, stdDev, p25, p50, p75, p90, p95, p99, p999, p100


//Cosine sim on single users with topic array saved from above
MATCH (p1:Card {cardId:83074})
MATCH (p2:Card) WHERE p2 <> p1 
RETURN p1.cardId as from, p2.cardId as to, algo.similarity.cosine(p1.topic_array, p2.topic_array) AS similarity
ORDER BY similarity DESC LIMIT 1000 



// Check for arrays with connection to beauty
MATCH (c:Card)-[:POSTED_ABOUT]->(img:ImageType)
WHERE img.cat = 0
RETURN c.cardId, c.image_array LIMIT 10





// ADD the Following to TENSORFLOW branch--------------------------------------------------------


//Create zero count relationships with every category
CALL apoc.periodic.iterate(
"MATCH (c:Card), (img:Imagetype) RETURN c, img",
"MERGE (c)-[r:IMAGE_TOPIC]->(img) ON CREATE SET r.count = 0", {batchSize:5000})

// This WORKED!? 
MATCH (c:Card), (img:Imagetype)
OPTIONAL MATCH (c)-[rel:IMAGE_TOPIC]->(img)
WITH {id:c.cardId, weights: collect(coalesce(rel.count, algo.NaN()))} as userData
WITH collect(userData) as data
UNWIND data as x
MERGE (c:Card {cardId:x.id}) ON MATCH SET c.image_array = x.weights


//Euc sim with topic array saved from above but slow
MATCH (p1:Card {cardId:22079})
MATCH (p2:Card) WHERE p2 <> p1 
RETURN p1.cardId as from, p2.cardId as to, algo.similarity.euclidean(p1.image_array, p2.image_array) AS similarity
ORDER BY similarity ASC LIMIT 10


//Write Cosine Similarity between the top 100 for each user. 
WITH "MATCH (person:Card)-[r:IMAGE_TOPIC]->(img)
      RETURN id(person) AS item, id(img) AS category, r.count AS weight" AS query
CALL algo.similarity.cosine(query, {
  graph: 'cypher', topK: 1000, similarityCutoff: 0.2, write:true, writeRelationshipType:"imageSim", writeProperty:"score"
})
YIELD nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, stdDev, p95
RETURN nodes, similarityPairs, write, writeRelationshipType, writeProperty, min, max, mean, p95



//Load image relationships on the big post file
CALL apoc.periodic.iterate("CALL apoc.load.json('file:///posts_250k.json') YIELD value", 
"MATCH (c:Card {card_id:toInt(value.cardID)}), (img:Imagetype {cat:toInt(value.cat)}) 
MERGE (c)-[r:IMAGE_TOPIC]->(img) ON CREATE SET r.count = 1 ON MATCH SET r.count = r.count + 1", {batchSize:10000})




// Links for url uploads: 
// Categories
https://raw.githubusercontent.com/xxx/category_list.csv
// Image Classifications
https://raw.githubusercontent.com/xxx/photo_cat.csv
//250k photo classifications
https://raw.githubusercontent.com/xxx/data.json
//Card data


//GRAPHENE Build in one shot:-----------------------------------------------------------------------------------------
LOAD CSV WITH HEADERS
FROM 'https://raw.githubusercontent.com/xxx/category_list.csv'
AS line
MERGE(c: Category {cat: line.Cat})
ON CREATE SET c.description=line.Desc;
LOAD CSV WITH HEADERS
FROM 'https://raw.githubusercontent.com/xxx/photo_cat.csv'
AS line
MERGE(c: Imagetype {cat: toInt(line.cat)})
ON CREATE SET c.description=line.description;
WITH 'https://something!!! ' as URL
CALL apoc.load.json(URL) YIELD line 
MERGE (c:Card {cardId:line.id})
WITH line.tags as cats  
MATCH (c:Card), (cat:Category)
WHERE c.cardId = line.id AND cat.cat in cats
MERGE (c)-[:CONTENT_TOPIC]->(cat);
MATCH (cat:Category)
WITH cat ORDER BY cat.cat
WITH collect(cat) AS cats
MATCH (p:Card)
SET p.topic_array = algo.ml.oneHotEncoding(cats, [(p)-[:CONTENT_TOPIC]->(cat) | cat]);
CALL apoc.periodic.iterate("
WITH 'https://raw.githubusercontent.com/xxx/data.json' as URL
CALL apoc.load.json(URL) YIELD value return value",
"MATCH (c:Card {cardId:value.cardId}), (img:Imagetype {cat:value.cat}) 
MERGE (c)-[r:IMAGE_TOPIC]->(img) 
ON CREATE SET r.count = 1
ON MATCH SET r.count = r.count + 1", {batchSize:10000});
CALL apoc.periodic.iterate(
"MATCH (c:Card), (img:Imagetype) RETURN c, img",
"MERGE (c)-[r:IMAGE_TOPIC]->(img) ON CREATE SET r.count = 0", {batchSize:5000});
MATCH (c:Card), (img:Imagetype)
OPTIONAL MATCH (c)-[rel:IMAGE_TOPIC]->(img)
WITH {id:c.cardId, weights: collect(coalesce(rel.count, algo.NaN()))} as userData
WITH collect(userData) as data
UNWIND data as x
MERGE (c:Card {cardId:x.id}) ON MATCH SET c.image_array = x.weights;


//NLP Write Similarity:
MATCH (c:With_nlp) WITH {item:id(c), weights: c.nlp_array} as userData WITH collect(userData) as data
CALL algo.similarity.cosine(data, {topK:1000, similarityCutoff: 0.7, write:true, writeRelationshipType:'NLP_SIMILARITY', writeProperty:'score', writeBatchSize:20000}) YIELD write, mean RETURN write, mean


MATCH (c:Card) WHERE reduce(total=0, number in c.image_array | total + number) < 5 SET c:With_photos