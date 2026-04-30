from __future__ import annotations

from http.server import ThreadingHTTPServer

from supportbot_web.config import DEFAULT_ARTIFACT_DIR, DEFAULT_HOST, DEFAULT_PORT
from supportbot_web.http_handler import SupportBotHandler


def main() -> None:
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), SupportBotHandler)
    url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    print(f"Support Bot v1 test arayuzu: {url}")
    print(f"Artifact: {DEFAULT_ARTIFACT_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()

