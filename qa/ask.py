import json
import faiss
import boto3
import numpy as np
from pathlib import Path

REGION = "us-east-1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
CHAT_MODEL_ID = "amazon.nova-lite-v1:0"

index_path = Path("qa/index.faiss")
ids_path = Path("qa/index_ids.json")
corpus_path = Path("qa/corpus.json")

if not index_path.exists() or not ids_path.exists() or not corpus_path.exists():
    raise SystemExit("Missing index artifacts. Run: python qa/build_corpus.py && python qa/embed_index.py")

index = faiss.read_index(str(index_path))
ids = json.loads(ids_path.read_text())
docs = json.loads(corpus_path.read_text())

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

def ask_once():
    q = input("\nAsk a lineage question: ").strip()
    if not q:
        print("Empty question.")
        return

    q_vec = np.array([embed(q)], dtype="float32")
    D, I = index.search(q_vec, 5)

    context = "\n".join(docs[i]["text"] for i in I[0])

    prompt = f"""You are a data lineage assistant.
Answer the QUESTION using ONLY the CONTEXT below. If the answer is not in context, say you don't know.

CONTEXT:
{context}

QUESTION:
{q}
"""

    resp = client.converse(
        modelId=CHAT_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 450, "temperature": 0}
    )

    print("\nANSWER:\n")
    print(resp["output"]["message"]["content"][0]["text"])

if __name__ == "__main__":
    ask_once()
