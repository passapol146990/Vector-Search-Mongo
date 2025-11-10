from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("intfloat/multilingual-e5-base")

mongo = MongoClient("")
db = mongo[""]
col = db[""]

def addEmbedding(modelV):
    products = list(col.find({"embedding": {"$exists": False}}))
    for p in products:
        emb = modelV.encode(p["productName"], convert_to_tensor=True).tolist()
        col.update_one({"_id": p["_id"]}, {"$set": {"embedding": emb}})
    print(f"✅ add embedding {len(products)} success")
# addEmbedding(model)

def search_products(query, limit=5):
    q_emb = model.encode(query, convert_to_tensor=True)
    all_products = list(col.find({}, {"productName": 1, "embedding": 1, "price": 1}))
    results = []

    for p in all_products:
        emb = torch.tensor(p["embedding"])
        score = util.cos_sim(q_emb, emb).item()
        results.append((score, p))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:limit]

for score, product in search_products("keyboard"):
    print(f"{score:.3f} | {product['productName']} - {product['price']}")
