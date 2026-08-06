#!/usr/bin/env python3
"""Self-hosted AI-text detector service for the DETECTOR_URL hook.

POST / with {"text": "..."} returns {"probability": 0.0-1.0} or
{"probability": null} when the text has too little prose to score (the
pipeline then skips the DETECTOR_SCORE rule). Stdlib HTTP server; the
model loads once at startup.

Usage: detector_server.py [bind-addr]   (default 127.0.0.1:9310)
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector_common import extract_prose, load_model, usable

_, _, SCORE = load_model()
print("detector model loaded", file=sys.stderr)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        try:
            body = json.loads(self.rfile.read(length))
            prose = extract_prose(str(body.get("text", "")))
            prob = round(SCORE(prose), 6) if usable(prose) else None
        except (json.JSONDecodeError, ValueError):
            self.send_response(400)
            self.end_headers()
            return
        payload = json.dumps({"probability": prob}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def main():
    addr = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:9310"
    host, port = addr.rsplit(":", 1)
    HTTPServer((host, int(port)), Handler).serve_forever()


if __name__ == "__main__":
    main()
