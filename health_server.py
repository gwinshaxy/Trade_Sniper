import os
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = int(os.getenv("PORT", 10000))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("health_server")

class HealthCheckHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP Handler for Render health checks."""
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "healthy", "service": "Bybit Futures Trading Bot System"}')

    def log_message(self, format, *args):
        return

def run():
    logger.info(f"Starting standalone Health Check server on 0.0.0.0:{PORT}...")
    server = HTTPServer(("0.0.0.0", PORT), HealthCheckHandler)
    server.serve_forever()

if __name__ == "__main__":
    run()