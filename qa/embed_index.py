import json
import faiss
import boto3
import numpy as np
from pathlib import Path

CORPUS_PATH = Path("qa/corpus.json")
INDEX_PATH = Path("qa/index.faiss")
IDS_PATH = Path("qa/index_ids.json")

REGION = "us-east-1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

if not CORPUS_PATH.exists():
    raise SystemExit("qa/corpus.json not found. Run: python qa/build_corpus.py")

docs = json.loads(CORPUS_PATH.read_text())
if not docs:
    raise SystemExit("Corpus is empty. Nothing to index.")

client = boto3.client("bedrock-runtime", region_name=REGION)

def embed(text: str):
    body = {"inputText": text}
    resp = client.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(resp["body"].read())["embedding"]

embeddings = []
ids = []

for d in docs:
    vec = embed(d["text"])
    embeddings.append(vec)
    ids.append(d["id"])

dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype="float32"))

faiss.write_index(index, str(INDEX_PATH))
IDS_PATH.write_text(json.dumps(ids, indent=2))

print(f"Built FAISS index: {INDEX_PATH} (vectors={len(ids)}, dim={dim})")
print(f"Wrote ids: {IDS_PATH}")
