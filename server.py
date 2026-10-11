"""
Lightweight HTTP Server for the BODMAS Tutor Agent.
Built using Python's standard library http.server for minimal footprint and maximum speed.
"""

import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

from harness.config import DEFAULT_MODEL, SUPPORTED_MODELS, get_api_key
from harness.pipeline import solve_bodmas_pipeline
from harness.eval import run_evaluation

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class BodmasRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def _send_json_response(self, status_code: int, data: dict):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            key_configured = bool(get_api_key())
            self._send_json_response(200, {
                "status": "healthy",
                "default_model": DEFAULT_MODEL,
                "supported_models": SUPPORTED_MODELS,
                "api_key_configured": key_configured,
            })
            return

        # Serve static web files
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            self._send_json_response(400, {"error": "Invalid JSON request body"})
            return

        if parsed.path == "/api/solve":
            equation = payload.get("equation", "").strip()
            if not equation:
                self._send_json_response(400, {"error": "Equation is required"})
                return

            tutor_temp = float(payload.get("tutor_temperature", 0.4))
            model_id = payload.get("model_id", DEFAULT_MODEL)
            api_key = payload.get("api_key")

            result = solve_bodmas_pipeline(
                raw_equation=equation,
                tutor_temperature=tutor_temp,
                model_id=model_id,
                api_key=api_key,
            )
            self._send_json_response(200, result)
            return

        elif parsed.path == "/api/eval":
            model_id = payload.get("model_id", DEFAULT_MODEL)
            api_key = payload.get("api_key")
            eval_results = run_evaluation(model_id=model_id, api_key=api_key)
            self._send_json_response(200, eval_results)
            return

        self._send_json_response(404, {"error": "Endpoint not found"})


def run_server(port: int = 8000):
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, BodmasRequestHandler)
    print(f"\n==================================================")
    print(f" ▮ BODMAS TUTOR SERVER RUNNING")
    print(f" ▮ Local URL: http://localhost:{port}")
    print(f" ▮ Model: {DEFAULT_MODEL}")
    print(f"==================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
