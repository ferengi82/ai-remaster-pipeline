from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_remaster_gui import references, state
from ai_remaster_gui.manifests import read_manifest, write_manifest_details


class ReferenceScrubTests(unittest.TestCase):
    def test_extract_reference_frame_persists_selected_frame(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(log=[], settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                folder = Path(tmp_text)
                source = folder / "source.mp4"
                source.write_bytes(b"video placeholder")
                manifest = folder / "shots.csv"
                write_manifest_details(
                    manifest,
                    references.rel(source),
                    ["start", "end", "source_reference", "color_reference"],
                    [{
                        "start": "00:00:00.000",
                        "end": "00:00:02.000",
                        "source_reference": "",
                        "color_reference": "",
                    }],
                )

                fake_result = SimpleNamespace(returncode=0, stderr="", stdout="")
                with (
                    mock.patch.object(references, "local_tool", return_value="ffmpeg"),
                    mock.patch.object(references, "ffprobe_info", return_value={"frame_rate": "25.000 fps"}),
                    mock.patch.object(references.subprocess, "run", return_value=fake_result),
                ):
                    result = references.extract_reference_frame(references.rel(manifest), 0, 1.24)

                row = read_manifest(manifest)[0]
                self.assertEqual(row["selected_frame"], "31")
                self.assertEqual(result["selected_frame"], "31")
                self.assertEqual(row["source_reference"], result["source_reference"])
        finally:
            state.APP = previous_app


if __name__ == "__main__":
    unittest.main()
