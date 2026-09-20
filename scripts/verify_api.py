"""Real HTTP smoke test against an already running final service."""

import argparse
import json
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8765")
args = parser.parse_args()


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        args.url + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, None


assert request("/health")[0] == 200
status, info = request("/model-info")
assert status == 200 and len(info["classes"]) == 77
for text in ["My card has not arrived yet", ["card", "💳世界", "reset my pin"]]:
    status, body = request("/predict", {"text": text})
    assert status == 200
    for row in body["predictions"]:
        assert 0 <= row["confidence"] <= 1 and row["priority"] is None
        assert (row["category"] is None) == row["requires_human"]
for text in ["", " ", "x" * 4001, 12, [], ["card"] * 65]:
    assert request("/predict", {"text": text})[0] == 422
print("10 real HTTP checks passed against final 77-class service")
