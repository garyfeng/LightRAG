# Instructinos for local testing with docker-compose
----

# Dockerfile for LightRAG

We created [Dockerfile](./Dockerfile) for LightRAG, using Python 3.11-slim as the basis. 

Notes:
- The starting script [test.py](./test.py) is modified from [test_neo4j.py](./test_neo4j.py), switching to using Azure OpenAI 
- Azure OpenAI configuration is in the [.env](./.env) file.

# Docker-compose using Neo4j

The [docker-compose.yml](./docker-compose.yml) launches 