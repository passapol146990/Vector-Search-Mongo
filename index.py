from dotenv import load_dotenv
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer

load_dotenv()

username = os.getenv("MONGO_USER")
password = os.getenv("MONGO_PASS")

uri = f"mongodb+srv://{username}:{password}@cluster0.aitteib.mongodb.net/?appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

db = client["vector_db"]
col = db["products"]

model = SentenceTransformer("intfloat/multilingual-e5-base")

sample = col.find_one({}, {"embedding": 1})
embed = model.encode("คีร์ร์บอร์ด", convert_to_tensor=False).tolist()

pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": embed,
            "numCandidates": 100,
            "limit": 5
        }
    }
]

results = list(col.aggregate(pipeline))

for r in results:
    name = r.get("productName", "N/A")
    link = r.get("productLink", "N/A")
    print(name,link)

