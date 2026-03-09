# Trademarkia Semantic Search

Semantic search system over 20 Newsgroups dataset with fuzzy clustering and semantic cache.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Pipeline (first time only)
```bash
python pipeline.py
```

## Start API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Semantic search with cache |
| GET | `/cache/stats` | Cache statistics |
| DELETE | `/cache` | Flush cache |
| GET | `/cluster/stats` | Cluster analysis |
| GET | `/health` | Health check |

## Test
Visit `http://localhost:8000/docs` for Swagger UI.

## Docker
#Live at https://hub.docker.com/r/raghavan004/trademarkia-search
```bash
docker-compose up --build
```