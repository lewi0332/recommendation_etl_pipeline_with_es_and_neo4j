# Flask API for individual reccommendations

This folder contains the 2 components to run a simple flask app that will allow for
gather recommended users given a targe user. 

- `main.py` - the primary flask app that sets up a route to get recommendations
    - route at `/recommend/` requires 3 arguments:
      - card_id: int
            The card_id of the user to get recommendations for
      - min: int
            The minimum number of followers the user should have
      - max: int
            The maximum number of followers the user should have
    - full url example `your-ip/recommend?card_id=1234&min=1000&max=100000`
      - This would return the 10 most similar card id's and the similarity scores for each
- `graph_recommend.py` - the function to call the Neo4j db and calculate most similar users



The method I used to run this in a beta test:

```
ssh first_ec2_ip

ssh internal_secure.ec2.internal

tmux a

sudo /opt/conda/envs/recommenderEnv/bin/python main.py

ctrl b + d to exit the tmux session
```

In beta, this allow for a dev test on the api only from machines 
inside the VPC and would also prohibit other machines from connecting to the DB

