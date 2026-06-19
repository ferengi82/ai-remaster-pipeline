"""Tests for the upscale logging helpers and the RunPod auto-shutdown control."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import upscale_video as uv  # noqa: E402

from ai_remaster_gui import pod_control  # noqa: E402


class FfmpegQuietingTests(unittest.TestCase):
    def test_quiet_flags_injected_by_default(self) -> None:
        captured: list[list[str]] = []
        with mock.patch.object(uv, "_FFMPEG_VERBOSE", False), \
                mock.patch.object(uv.subprocess, "run", side_effect=lambda c, **k: captured.append(c)):
            uv.run_ffmpeg(["ffmpeg", "-y", "-i", "in.mp4", "out.mp4"])
        self.assertEqual(captured[0][:4], ["ffmpeg", "-hide_banner", "-loglevel", "error"])
        self.assertIn("-nostats", captured[0])
        # Original arguments are preserved after the injected flags.
        self.assertEqual(captured[0][-3:], ["-i", "in.mp4", "out.mp4"])

    def test_verbose_env_keeps_original_command(self) -> None:
        captured: list[list[str]] = []
        with mock.patch.object(uv, "_FFMPEG_VERBOSE", True), \
                mock.patch.object(uv.subprocess, "run", side_effect=lambda c, **k: captured.append(c)):
            uv.run_ffmpeg(["ffmpeg", "-y", "out.mp4"])
        self.assertEqual(captured[0], ["ffmpeg", "-y", "out.mp4"])

    def test_existing_loglevel_not_overridden(self) -> None:
        captured: list[list[str]] = []
        with mock.patch.object(uv, "_FFMPEG_VERBOSE", False), \
                mock.patch.object(uv.subprocess, "run", side_effect=lambda c, **k: captured.append(c)):
            uv.run_ffmpeg(["ffmpeg", "-loglevel", "info", "out.mp4"])
        self.assertEqual(captured[0], ["ffmpeg", "-loglevel", "info", "out.mp4"])

    def test_nonzero_exit_raises(self) -> None:
        import subprocess
        with mock.patch.object(uv.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            with self.assertRaises(subprocess.CalledProcessError):
                uv.run_ffmpeg(["ffmpeg", "out.mp4"])


class GpuLabelTests(unittest.TestCase):
    def test_port_maps_to_gpu_index(self) -> None:
        self.assertEqual(uv.gpu_label("http://127.0.0.1:8188"), "GPU0 :8188")
        self.assertEqual(uv.gpu_label("http://127.0.0.1:8190"), "GPU2 :8190")

    def test_non_instance_url_falls_back_to_url(self) -> None:
        self.assertEqual(uv.gpu_label("http://example.com:9000"), "GPU812 :9000")
        self.assertEqual(uv.gpu_label("not a url"), "not a url")


class FmtDurationTests(unittest.TestCase):
    def test_formats(self) -> None:
        self.assertEqual(uv.fmt_duration(0), "0s")
        self.assertEqual(uv.fmt_duration(18.3), "18s")
        self.assertEqual(uv.fmt_duration(214), "3m34s")
        self.assertEqual(uv.fmt_duration(4332), "1h12m")


class ShutdownPodTests(unittest.TestCase):
    def test_disabled_action_is_noop(self) -> None:
        with mock.patch.dict("os.environ", {"RUNPOD_POD_ID": "abc"}, clear=False):
            self.assertFalse(pod_control.shutdown_pod(""))
            self.assertFalse(pod_control.shutdown_pod("nonsense"))

    def test_missing_pod_id_skips(self) -> None:
        logs: list[str] = []
        with mock.patch.dict("os.environ", {}, clear=True):
            result = pod_control.shutdown_pod("stop", log=logs.append)
        self.assertFalse(result)
        self.assertTrue(any("RUNPOD_POD_ID" in line for line in logs))

    def test_api_key_path_calls_graphql_stop(self) -> None:
        calls: list[str] = []

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"data": {"podStop": {"id": "abc", "desiredStatus": "EXITED"}}}'

        def fake_urlopen(request, timeout=0):
            calls.append(request.data.decode("utf-8"))
            return FakeResp()

        env = {"RUNPOD_POD_ID": "abc", "RUNPOD_API_KEY": "key"}
        with mock.patch.dict("os.environ", env, clear=True), \
                mock.patch.object(pod_control.urllib.request, "urlopen", side_effect=fake_urlopen), \
                mock.patch.object(pod_control.subprocess, "run") as cli:
            result = pod_control.shutdown_pod("stop", log=lambda _m: None)
        self.assertTrue(result)
        self.assertIn("podStop", calls[0])
        cli.assert_not_called()  # API path used; CLI not invoked

    def test_falls_back_to_runpodctl_without_api_key(self) -> None:
        env = {"RUNPOD_POD_ID": "abc"}
        with mock.patch.dict("os.environ", env, clear=True), \
                mock.patch.object(pod_control.subprocess, "run") as cli:
            result = pod_control.shutdown_pod("terminate", log=lambda _m: None)
        self.assertTrue(result)
        cli.assert_called_once()
        self.assertEqual(cli.call_args.args[0], ["runpodctl", "remove", "pod", "abc"])

    def test_api_error_falls_back_to_cli(self) -> None:
        env = {"RUNPOD_POD_ID": "abc", "RUNPOD_API_KEY": "key"}
        with mock.patch.dict("os.environ", env, clear=True), \
                mock.patch.object(pod_control.urllib.request, "urlopen", side_effect=OSError("boom")), \
                mock.patch.object(pod_control.subprocess, "run") as cli:
            result = pod_control.shutdown_pod("stop", log=lambda _m: None)
        self.assertTrue(result)
        cli.assert_called_once()
        self.assertEqual(cli.call_args.args[0], ["runpodctl", "stop", "pod", "abc"])


if __name__ == "__main__":
    unittest.main()
