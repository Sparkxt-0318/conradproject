"""Serves the field app's files, for trying it on the laptop or handing it to a phone on the
same network.

A phone only gets GPS and offline start-up from a secure address (https, or localhost on the
phone itself). For field use, put the static folder on any https web host once; this server
is for the desk.
"""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"


class _Handler(SimpleHTTPRequestHandler):
    package: bytes | None = None

    def do_GET(self):
        if self.path.split("?")[0] == "/package.json":
            if self.package is None:
                self.send_error(404, "no field package is being served")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.package)))
            self.end_headers()
            self.wfile.write(self.package)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *args):
        pass


def make_server(host: str = "127.0.0.1", port: int = 8000, package_path=None) -> ThreadingHTTPServer:
    """A server for the app; with package_path, the app loads that field package by itself."""
    handler = type("Handler", (_Handler,), {"package": Path(package_path).read_bytes() if package_path else None})
    return ThreadingHTTPServer((host, port), partial(handler, directory=str(STATIC)))
