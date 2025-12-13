import json
import boto3

cfg = json.load(open("config.json"))
client = boto3.client("bedrock-runtime", region_name=cfg["region"])

resp = client.converse(
    modelId=cfg["model_id"],
    messages=[
        {"role": "user", "content": [{"text": "Reply with exactly: Bedrock runtime works"}]}
    ],
    inferenceConfig={"maxTokens": 60, "temperature": 0}
)

print(resp["output"]["message"]["content"][0]["text"].strip())
