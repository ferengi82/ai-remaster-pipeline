"""Security regression tests for the GUI HTTP layer.

These guard the protections that keep the localhost GUI from being abused by a malicious web page
the user happens to have open: arbitrary file reads, DNS rebinding, CSRF, and SSRF. They drive the
request handler directly with hand-built headers rather than over a real socket.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_remaster_gui import config, http_handler
from ai_remaster_gui import server  # noqa: F401  (imported for its bind_context side effect)
from ai_remaster_gui.paths import resolve_served


class ResolveServedTests(unittest.TestCase):
    """The path allowlist that backs /media, /api/logfile and /api/media-status."""

    def test_path_inside_project_is_allowed(self) -> None:
        self.assertIsNotNone(resolve_served("README.md"))

    def test_absolute_path_outside_project_is_rejected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt") as handle:
            self.assertIsNone(resolve_served(handle.name))

    def test_parent_directory_traversal_is_rejected(self) -> None:
        self.assertIsNone(resolve_served("../../etc/passwd"))
        self.assertIsNone(resolve_served("..\\..\\Windows\\win.ini"))

    def test_empty_path_is_rejected(self) -> None:
        self.assertIsNone(resolve_served(""))

    def test_selected_source_folder_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "clip.mp4"
            sibling = Path(folder) / "clip.preview.png"
            # A file next to the chosen source resolves only because that folder is passed in.
            self.assertIsNone(resolve_served(str(sibling)))
            self.assertIsNotNone(resolve_served(str(sibling), [str(source)]))


class _RecordingHandler(http_handler.Handler):
    """A Handler that captures responses instead of writing them to a socket."""

    def __init__(self, headers: dict[str, str], path: str = "/") -> None:  # noqa: D401 - test stub
        self.headers = headers
        self.path = path
        self.responses: dict[str, object] = {}

    def send_error(self, code, message=None, explain=None):  # noqa: D102
        self.responses["error"] = code

    def send_json(self, payload):  # noqa: D102
        self.responses["json"] = payload

    def send_media(self, path):  # noqa: D102
        self.responses["media"] = path

    def send_static(self, *args, **kwargs):  # noqa: D102
        self.responses["static"] = args

    def read_json(self):  # noqa: D102
        return {}


LOCAL = {"Host": "127.0.0.1:8765"}


class HostHeaderTests(unittest.TestCase):
    """Reject requests whose Host header is not loopback (DNS-rebinding defence)."""

    def test_remote_host_blocks_get(self) -> None:
        handler = _RecordingHandler({"Host": "evil.com"}, "/api/state")
        handler.do_GET()
        self.assertEqual(handler.responses.get("error"), 403)

    def test_remote_host_blocks_post(self) -> None:
        handler = _RecordingHandler({"Host": "evil.com"}, "/api/stop")
        handler.do_POST()
        self.assertEqual(handler.responses.get("error"), 403)

    def test_loopback_host_passes_the_gate(self) -> None:
        # An allowed request with an unknown path falls through to a 404, proving the gate let it by.
        handler = _RecordingHandler(dict(LOCAL), "/api/does-not-exist")
        handler.do_POST()
        self.assertEqual(handler.responses.get("error"), 404)


class OriginTests(unittest.TestCase):
    """Reject cross-origin POSTs (CSRF defence)."""

    def test_cross_origin_post_blocked(self) -> None:
        handler = _RecordingHandler({**LOCAL, "Origin": "https://evil.com"}, "/api/stop")
        handler.do_POST()
        self.assertEqual(handler.responses.get("error"), 403)

    def test_cross_origin_referer_blocked(self) -> None:
        handler = _RecordingHandler({**LOCAL, "Referer": "https://evil.com/x"}, "/api/stop")
        handler.do_POST()
        self.assertEqual(handler.responses.get("error"), 403)

    def test_same_origin_post_allowed(self) -> None:
        handler = _RecordingHandler({**LOCAL, "Origin": "http://127.0.0.1:8765"}, "/api/does-not-exist")
        handler.do_POST()
        self.assertEqual(handler.responses.get("error"), 404)


class MediaEndpointTests(unittest.TestCase):
    """The reported arbitrary-file-read endpoints are confined to the allowlist."""

    def test_media_rejects_path_outside_project(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            handler = _RecordingHandler(dict(LOCAL), "/media?path=" + handle.name)
            handler.do_GET()
        self.assertEqual(handler.responses.get("error"), 404)
        self.assertNotIn("media", handler.responses)

    def test_media_serves_path_inside_project(self) -> None:
        handler = _RecordingHandler(dict(LOCAL), "/media?path=README.md")
        handler.do_GET()
        self.assertIn("media", handler.responses)
        self.assertEqual(Path(handler.responses["media"]).name, "README.md")

    def test_logfile_outside_project_returns_empty(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as handle:
            handle.write(b"secret log contents")
            secret = handle.name
        try:
            handler = _RecordingHandler(dict(LOCAL), "/api/logfile?path=" + secret)
            handler.do_GET()
        finally:
            Path(secret).unlink()
        self.assertEqual(handler.responses.get("json"), {"text": ""})

    def test_media_status_outside_project_reports_missing(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            handler = _RecordingHandler(dict(LOCAL), "/api/media-status?path=" + handle.name)
            handler.do_GET()
        self.assertFalse(handler.responses["json"]["exists"])


class ComfyProxyTests(unittest.TestCase):
    """The ComfyUI connectivity probe must not become an SSRF / file-read primitive."""

    def test_non_http_scheme_rejected(self) -> None:
        handler = _RecordingHandler(dict(LOCAL), "/api/comfy?url=file:///etc/passwd")
        handler.do_GET()
        payload = handler.responses["json"]
        self.assertFalse(payload["ok"])
        self.assertIn("http", payload["error"])


if __name__ == "__main__":
    unittest.main()
