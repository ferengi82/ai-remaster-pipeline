from __future__ import annotations

import copy
import csv
import json
import shutil
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import unittest
import sys
import argparse
import importlib.util
import os
import zipfile
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import comfy_api  # noqa: E402
import audio_models  # noqa: E402
import colorize_video  # noqa: E402
import create_audio_track  # noqa: E402
import generate_single_reference  # noqa: E402
import guide_frame_utils  # noqa: E402
import edit_reference_image  # noqa: E402
import openai_generate_reference  # noqa: E402
import outpaint_video  # noqa: E402
import prepare_outpaint_input  # noqa: E402
import qwen_colorize_references  # noqa: E402
import upscale_video  # noqa: E402

from ai_remaster_gui import app
from ai_remaster_gui import config
from ai_remaster_gui import outpaint_guides
from ai_remaster_gui import sam_masks
from ai_remaster_gui import server


class GuiSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._settings = copy.deepcopy(app.APP.settings)
        # Default the soundtrack phase off so stage-order / upscale-chaining tests are not
        # affected by whatever add_soundtrack happens to be in the loaded settings.
        app.APP.settings.setdefault("global", {})["add_soundtrack"] = "false"

    def tearDown(self) -> None:
        app.APP.settings = self._settings

    def test_source_resolver_accepts_ascii_pipe_for_full_width_pipe_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            real = folder / "King Kong Scene Pack ｜ King Kong [0JgMh4I2UjY].mp4"
            real.write_bytes(b"not a real video")
            typed = folder / "King Kong Scene Pack | King Kong [0JgMh4I2UjY].mp4"

            self.assertEqual(app.resolve_video_source(str(typed)), real)

    def test_run_all_waits_for_stage_hydration_before_next_stage(self) -> None:
        class DoneProcess:
            returncode = 0

            def poll(self):
                return 0

        stages = [stage for stage in app.STAGES if stage.key in {"outpaint", "shots"}]
        seen: list[tuple[str, str]] = []

        def fake_run_stage(stage_key: str) -> tuple[bool, str]:
            seen.append((stage_key, app.APP.settings.get("shots", {}).get("outpainted_video", "")))
            app.APP.process = DoneProcess()
            app.APP.running_stage_key = stage_key
            if stage_key == "outpaint":
                def finish_hydration() -> None:
                    time.sleep(0.1)
                    app.APP.settings.setdefault("shots", {})["outpainted_video"] = "intermediate/outpainted/movie.mp4"
                    app.APP.running_stage_key = ""

                threading.Thread(target=finish_hydration).start()
            else:
                app.APP.running_stage_key = ""
            return True, "started"

        with mock.patch.object(app.APP, "active_stages", return_value=tuple(stages)), mock.patch.object(app.APP, "run_stage", side_effect=fake_run_stage):
            app.APP._run_all_worker()

        self.assertEqual([key for key, _ in seen], ["outpaint", "shots"])
        self.assertEqual(seen[1][1], "intermediate/outpainted/movie.mp4")

    def test_deterministic_outpaint_output_path_uses_selected_source(self) -> None:
        app.APP.settings.setdefault("outpaint", {}).update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        output = app.outpaint_output_for("input/My Source.mp4", "16:9", "720")

        # Identity-keyed short name: <sourceword>_<tag>_<key>.mp4 under intermediate/outpainted/.
        self.assertTrue(output.startswith("intermediate/outpainted/My_outpaint_"), output)
        self.assertTrue(output.endswith(".mp4"))
        # The GUI locator and the producer script must name the file identically (no drift).
        args = argparse.Namespace(crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, outpaint_all_black_regions=False)
        producer = outpaint_video.default_output(app.resolve_video_source("input/My Source.mp4"), "16:9", 720, args)
        self.assertEqual(Path(output).name, producer.name)

    def test_outpaint_ltx_working_paths_use_model_safe_size(self) -> None:
        app.APP.settings.setdefault("outpaint", {}).update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        prepared = app.outpaint_prepared_for("input/My Source.mp4", app.APP.settings["outpaint"])
        manifest = app.outpaint_chunk_manifest_for("input/My Source.mp4", app.APP.settings["outpaint"])

        self.assertEqual(prepared.parent.name, "outpaint_prepared")
        self.assertTrue(prepared.name.startswith("My_prepared_"), prepared.name)
        self.assertTrue(manifest.startswith("manifests/outpaint_chunks/My_chunks_"), manifest)
        self.assertTrue(manifest.endswith(".csv"))
        # GUI and producer must agree on the prepared-canvas and outpaint output names.
        args = argparse.Namespace(crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, outpaint_all_black_regions=False)
        source = app.resolve_video_source("input/My Source.mp4")
        self.assertEqual(prepared.name, outpaint_video.prepared_for(source, "16:9", 720, args).name)
        self.assertEqual(
            Path(app.outpaint_output_for("input/My Source.mp4", "16:9", "720")).name,
            outpaint_video.default_output(source, "16:9", 720, args).name,
        )

    def test_source_height_outpaint_option_uses_video_height(self) -> None:
        app.APP.settings.setdefault("outpaint", {}).update(
            {
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )
        with mock.patch.object(server, "video_metrics", return_value={"height": 480}):
            output = app.outpaint_output_for("input/My Source.mp4", "16:9", "source")
        output_720 = app.outpaint_output_for("input/My Source.mp4", "16:9", "720")

        self.assertTrue(output.startswith("intermediate/outpainted/My_outpaint_"), output)
        # "Source height" (480 -> work 864x480) must differ from a fixed 720 (-> 1280x704).
        self.assertNotEqual(output, output_720)
        # Agrees with the producer at the resolved height (480).
        args = argparse.Namespace(crop_left=0, crop_right=0, crop_top=0, crop_bottom=0, outpaint_all_black_regions=False)
        producer = outpaint_video.default_output(app.resolve_video_source("input/My Source.mp4"), "16:9", 480, args)
        self.assertEqual(Path(output).name, producer.name)

    def test_outpaint_source_crop_preserves_original_source_scale(self) -> None:
        args = argparse.Namespace(
            delivery_width=1280,
            delivery_height=720,
            crop_left=0,
            crop_right=0,
            crop_top=270,
            crop_bottom=270,
            black_lift=0.018,
            gamma=1.06,
            outpaint_all_black_regions=False,
        )
        info = {"width": 1440, "height": 1080, "fps": 24.0}

        self.assertEqual(prepare_outpaint_input.source_placement_size(args, info, 1280, 704), (960, 360, 960, 352))

        filter_text = prepare_outpaint_input.build_filter(args, info, 1280, 704)

        self.assertIn("crop=w=1440:h=540:x=0:y=270,scale=w=960:h=360:flags=lanczos,scale=w=960:h=352:flags=lanczos", filter_text)
        self.assertNotIn("force_original_aspect_ratio=decrease", filter_text)

    def test_outpaint_all_black_regions_bypasses_source_lift(self) -> None:
        args = argparse.Namespace(
            delivery_width=1280,
            delivery_height=720,
            crop_left=0,
            crop_right=0,
            crop_top=0,
            crop_bottom=0,
            black_lift=0.018,
            gamma=1.06,
            outpaint_all_black_regions=True,
        )
        info = {"width": 960, "height": 720, "fps": 24.0}

        filter_text = prepare_outpaint_input.build_filter(args, info, 1280, 704)

        self.assertNotIn("lutrgb=", filter_text)
        self.assertIn("color=c=black:s=1280x704", filter_text)

    def test_outpaint_all_black_regions_changes_output_paths_and_command(self) -> None:
        app.APP.settings.setdefault("outpaint", {}).update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
                "outpaint_all_black_regions": "true",
            }
        )
        app.APP.settings["global"].update({"source": "input/My Source.mp4", "section_start": "0", "section_end": ""})

        output_black = app.outpaint_output_for("input/My Source.mp4", "16:9", "720")
        prepared_black = app.outpaint_prepared_for("input/My Source.mp4", app.APP.settings["outpaint"])
        command = app.APP.command_for("outpaint")
        # The all-black variant must be a distinct artifact from the protected-blacks variant.
        app.APP.settings["outpaint"]["outpaint_all_black_regions"] = "false"
        output_plain = app.outpaint_output_for("input/My Source.mp4", "16:9", "720")
        app.APP.settings["outpaint"]["outpaint_all_black_regions"] = "true"

        self.assertTrue(output_black.startswith("intermediate/outpainted/My_outpaint_"), output_black)
        self.assertNotEqual(output_black, output_plain)
        self.assertTrue(prepared_black.name.startswith("My_prepared_"), prepared_black.name)
        self.assertIn("--outpaint-all-black-regions", command)

    def test_portable_comfy_parent_resolves_to_inner_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            portable = Path(tmp_text)
            inner = portable / "ComfyUI"
            inner.mkdir()
            (inner / "main.py").write_text("# comfy\n", encoding="utf-8")

            self.assertEqual(config.resolve_comfy_dir(str(portable)), inner)

    def test_required_comfy_workflows_are_bundled(self) -> None:
        outpaint = app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json"
        qwen = app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json"

        for workflow in (outpaint, qwen):
            with self.subTest(workflow=workflow.name):
                self.assertTrue(workflow.exists(), f"Missing bundled workflow: {workflow}")
                json.loads(workflow.read_text(encoding="utf-8-sig"))

        self.assertEqual(server.default_qwen_workflow({}), app.rel(qwen))
        self.assertEqual(server.qwen_workflow_for({"workflow": "D:/missing/blueprints/Qwen Custom.json"}, {}), app.rel(qwen))
        self.assertEqual(
            server.qwen_workflow_for(
                {
                    "workflow": (
                        "D:/ComfyUI/venv/Lib/site-packages/"
                        "comfyui_workflow_templates_media_image/templates/image_qwen_image_edit_2511.json"
                    )
                },
                {},
            ),
            app.rel(qwen),
        )

    def test_required_custom_nodes_are_bundled(self) -> None:
        required = {
            "ComfyUI-LTXVideo": ("LTXVImgToVideoConditionOnly", "LTXAddVideoICLoRAGuide", "LTXVPreprocess"),
            "ComfyUI-GGUF": ("UnetLoaderGGUF",),
            "ComfyUI-VideoHelperSuite": ("VHS_LoadVideo", "VHS_VideoCombine"),
            "ComfyUI-FlashVSR_Ultra_Fast": ("FlashVSRNode",),
            "reference-video-colorization": ("DeepExColorVideoNode", "ColorMNetVideo"),
        }
        vendor_root = app.ROOT / "vendor" / "comfyui_custom_nodes"
        for folder, symbols in required.items():
            with self.subTest(folder=folder):
                package = vendor_root / folder
                self.assertTrue(package.is_dir(), f"Missing bundled custom node package: {package}")
                self.assertTrue((package / "LICENSE").exists(), f"Missing bundled custom node license: {package}")
                texts = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in package.rglob("*.py"))
                for symbol in symbols:
                    self.assertIn(symbol, texts)

    def test_wait_for_prompt_retries_transient_polling_errors(self) -> None:
        calls = {"count": 0}

        def fake_http_json(method, url, timeout=30):
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("Timed out waiting for ComfyUI")
            return {"prompt-id": {"status": {"completed": True}, "outputs": {}}}

        with mock.patch.object(comfy_api, "http_json", side_effect=fake_http_json), mock.patch.object(comfy_api.time, "sleep"):
            history = comfy_api.wait_for_prompt("http://127.0.0.1:8188", "prompt-id", 0.01, transient_timeout_seconds=30)

        self.assertEqual(calls["count"], 3)
        self.assertEqual(history["status"]["completed"], True)

    def test_outpaint_prompt_bypasses_unbundled_kj_padding_node(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))

        outpaint_video.bypass_optional_preview_nodes(workflow)
        outpaint_video.bypass_demo_padding_node(workflow)
        prompt = comfy_api.workflow_to_prompt(workflow, "5076")

        class_types = {node["class_type"] for node in prompt.values()}
        self.assertNotIn("ImagePadKJ", class_types)
        self.assertIn("VHS_LoadVideo", class_types)
        self.assertIn("VHS_VideoCombine", class_types)

    def test_outpaint_prompt_sent_to_ic_lora_guide_combines_global_and_chunk_suffix(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args(
            [
                "--source",
                "input/example.mp4",
                "--comfy-dir",
                str(app.ROOT),
                "--prompt",
                "outpaint with natural film grain",
                "--dry-run",
            ]
        )

        with (
            mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mp4"),
            mock.patch.object(outpaint_video, "copy_reference_frame_to_comfy_input", return_value="arp_outpaint/reference.png"),
            mock.patch.object(outpaint_video, "probe_video", return_value={"width": 1280, "height": 704, "frames": 24, "fps": 24.0}),
        ):
            prompt = outpaint_video.patch_workflow(
                args,
                workflow,
                app.ROOT / "prepared.mp4",
                app.ROOT,
                "arp_outpaint/test",
                outpaint_video.combine_prompt(args.prompt, "continue the wallpaper"),
                args.negative_prompt,
                42,
            )

        self.assertEqual(prompt["2483"]["inputs"]["text"], "outpaint with natural film grain. continue the wallpaper")
        self.assertEqual(prompt["5012"]["inputs"]["positive"], ["1241", 0])
        self.assertEqual(prompt["1241"]["inputs"]["positive"], ["2483", 0])

    def test_outpaint_conditioning_bypasses_resize_without_replacing_video_control(self) -> None:
        workflow = json.loads((app.ROOT / "workflows" / "outpaint_ltx" / "outpaint_LTX-IC.json").read_text(encoding="utf-8-sig"))
        args = outpaint_video.build_parser().parse_args(["--source", "input/example.mp4", "--comfy-dir", str(app.ROOT), "--dry-run"])

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            guide = Path(tmp_text) / "guide.png"
            guide.write_bytes(b"guide")
            with (
                mock.patch.object(outpaint_video, "copy_to_comfy_input", return_value="arp_outpaint/prepared.mp4"),
                mock.patch.object(outpaint_video, "copy_guide_image_to_comfy_input", return_value="arp_outpaint/guide_864x480.png"),
                mock.patch.object(outpaint_video, "probe_video", return_value={"width": 864, "height": 480, "frames": 24, "fps": 24.0}),
            ):
                prompt = outpaint_video.patch_workflow(
                    args,
                    workflow,
                    app.ROOT / "prepared.mp4",
                    app.ROOT,
                    "arp_outpaint/test",
                    args.prompt,
                    args.negative_prompt,
                    42,
                    guide,
                )

        self.assertEqual(prompt["3336"]["inputs"]["image"], ["2004", 0])
        self.assertEqual(prompt["5012"]["inputs"]["image"], ["5060", 0])
        self.assertEqual(prompt["2004"]["inputs"]["image"], "arp_outpaint/guide_864x480.png")

    def test_qwen_seed_guides_do_not_overwrite_existing_set_guides(self) -> None:
        args = argparse.Namespace(
            comfy_output_root="",
            comfy_dir=str(app.ROOT),
            qwen_workflow="workflow.json",
            qwen_masked_workflow="masked.json",
            comfy_url="http://127.0.0.1:8188",
            qwen_model_backend="gguf",
            qwen_gguf_model="model.gguf",
            qwen_prompt="Replace the black bars.",
            qwen_load_image_node_id="auto",
            qwen_save_node_id="auto",
            seed_sample_seconds=0.0,
            seed_shot_threshold=None,
            seed_min_shot_seconds=None,
            guide_strength=0.7,
            force=False,
        )
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            guide = folder / "existing.png"
            guide.write_bytes(b"guide")
            existing = [{"frame_idx": 0, "strength": 0.7, "image": app.rel(guide), "seed": True}]
            outpaint_video.write_chunk_manifest(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "24",
                        "guide_frames": json.dumps(existing),
                    }
                ],
            )

            with mock.patch.object(outpaint_video, "seed_guides", return_value={}) as seed_mock:
                rows = outpaint_video.apply_qwen_seed_guides(args, folder / "prepared.mp4", [(0, 0, 24)], manifest)

        self.assertEqual(json.loads(rows[0]["guide_frames"]), existing)
        self.assertEqual(seed_mock.call_args.kwargs["occupied_frame_idxs"], {0: {0}})

    def test_outpaint_command_uses_global_prompt(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "chunk_seconds": "20",
                "overlap_frames": "8",
                "prompt": "outpaint with restrained natural edges",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--prompt") + 1], "outpaint with restrained natural edges")

    def test_outpaint_command_falls_back_to_activation_prompt_when_global_prompt_blank(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "chunk_seconds": "20",
                "overlap_frames": "8",
                "prompt": "",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        command = app.APP.command_for("outpaint")

        self.assertEqual(command[command.index("--prompt") + 1], "outpaint")

    def test_outpaint_prompt_combiner_adds_sentence_separator_for_chunk_suffix(self) -> None:
        self.assertEqual(outpaint_video.combine_prompt("outpaint", "continue the room"), "outpaint. continue the room")
        self.assertEqual(outpaint_video.combine_prompt("outpaint.", "continue the room"), "outpaint. continue the room")
        self.assertEqual(outpaint_video.combine_prompt("", "continue the room"), "continue the room")

    def test_outpaint_manifest_sync_preserves_chunk_prompt_suffixes(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            outpaint_video.write_chunk_manifest(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "10",
                        "seed": "42",
                        "prompt_suffix": "stale per-chunk direction",
                    }
                ],
            )

            rows = outpaint_video.sync_chunk_manifest(manifest, [(0, 0, 10)], 24.0, folder, 42)

            self.assertEqual(rows[0]["prompt_suffix"], "stale per-chunk direction")

    def test_outpaint_manifest_sync_uses_offset_specific_chunk_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            outpaint_video.write_chunk_manifest(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "10",
                        "seed": "42",
                        "offset_x": "12",
                        "offset_y": "-4",
                    }
                ],
            )

            rows = outpaint_video.sync_chunk_manifest(manifest, [(0, 0, 10)], 24.0, folder, 42)

            self.assertIn("_ox+12_oy-4", rows[0]["prepared_path"])
            self.assertIn("_ox+12_oy-4", rows[0]["raw_path"])

    def test_colormnet_correlation_extension_install_is_opt_in(self) -> None:
        downloader_path = app.ROOT / "vendor" / "comfyui_custom_nodes" / "reference-video-colorization" / "colormnet" / "downloader.py"
        spec = importlib.util.spec_from_file_location("colormnet_downloader_under_test", downloader_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(module.optional_correlation_install_enabled())

        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"COLORMNET_INSTALL_CORRELATION_EXTENSION": value}, clear=True):
                self.assertTrue(module.optional_correlation_install_enabled())

    def test_single_reference_rejects_missing_source_before_qwen_startup(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            missing = Path(tmp_text) / "missing.png"
            output = Path(tmp_text) / "output.png"
            argv = [
                "generate_single_reference.py",
                "--source-image",
                str(missing),
                "--output",
                str(output),
                "--workflow",
                str(app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json"),
                "--dry-run",
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(generate_single_reference.qwen, "main_with_args") as qwen_main:
                with self.assertRaisesRegex(FileNotFoundError, "Reference source image not found"):
                    generate_single_reference.main()

            qwen_main.assert_not_called()

    def test_bundled_qwen_2511_subgraph_patches_to_gguf_by_default(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.png"
            source.write_bytes(b"placeholder")
            comfy_dir = folder / "comfy"
            (comfy_dir / "input").mkdir(parents=True)
            workflow = qwen_colorize_references.load_workflow(app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json")
            args = argparse.Namespace(comfy_dir=comfy_dir, model_backend="gguf", gguf_model="qwen-image-edit-2511-Q4_K_M.gguf")

            qwen_colorize_references.patch_qwen_model_backend(args, workflow)
            prompt = qwen_colorize_references.patch_workflow(args, workflow, source, folder / "output.png", "Colorize this image.")

            self.assertEqual(prompt["169"]["inputs"]["seed"], 1)
            self.assertEqual(prompt["169"]["inputs"]["control_after_generate"], "fixed")
            self.assertEqual(prompt["161"]["class_type"], "UnetLoaderGGUF")
            self.assertEqual(prompt["161"]["inputs"]["unet_name"], "qwen-image-edit-2511-Q4_K_M.gguf")

    def test_qwen_completion_copies_produced_image_to_requested_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.png"
            output = folder / "output.png"
            produced = folder / "comfy-output.png"
            workflow = folder / "workflow.json"
            manifest = folder / "manifest.csv"
            comfy_dir = folder / "comfy"
            comfy_output = folder / "comfy-output"
            source.write_bytes(b"source image bytes")
            produced.write_bytes(b"produced image bytes")
            workflow.write_text("{}", encoding="utf-8")
            comfy_dir.mkdir()
            comfy_output.mkdir()
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_reference", "color_reference"])
                writer.writeheader()
                writer.writerow({"source_reference": str(source), "color_reference": str(output)})

            args = qwen_colorize_references.build_parser().parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--workflow",
                    str(workflow),
                    "--comfy-dir",
                    str(comfy_dir),
                    "--comfy-output-root",
                    str(comfy_output),
                    "--model-backend",
                    "safetensors",
                    "--no-normalize-to-source-size",
                    "--force",
                ]
            )

            with (
                mock.patch.object(qwen_colorize_references, "ensure_qwen_image_edit_models"),
                mock.patch.object(qwen_colorize_references, "wait_for_comfy"),
                mock.patch.object(qwen_colorize_references, "patch_qwen_model_backend"),
                mock.patch.object(qwen_colorize_references, "patch_workflow", return_value={}),
                mock.patch.object(qwen_colorize_references, "queue_prompt", return_value="prompt-id"),
                mock.patch.object(qwen_colorize_references, "wait_for_prompt", return_value={}),
                mock.patch.object(qwen_colorize_references, "extract_output_files", return_value=[produced]),
                mock.patch.object(qwen_colorize_references, "newest_output", return_value=produced),
            ):
                self.assertEqual(qwen_colorize_references.main_with_args(args), 0)

            self.assertEqual(output.read_bytes(), b"produced image bytes")

    def test_outpaint_chunk_rows_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "10",
                    "start_seconds": "0.000000",
                    "end_seconds": "0.416667",
                    "seed": "42",
                    "prompt_suffix": "",
                    "prepared_path": "prepared.mp4",
                    "raw_path": "raw.mp4",
                }
            ]

            app.write_outpaint_chunk_rows(manifest, rows)

            self.assertEqual(app.read_outpaint_chunk_rows(manifest)[0]["raw_path"], "raw.mp4")

    def test_outpaint_chunk_rows_do_not_rewrite_identical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [{"chunk_index": "0", "start_frame": "0", "end_frame": "10", "seed": "42"}]

            app.write_outpaint_chunk_rows(manifest, rows)
            first_mtime = manifest.stat().st_mtime_ns
            app.write_outpaint_chunk_rows(manifest, rows)

            self.assertEqual(manifest.stat().st_mtime_ns, first_mtime)

    def test_outpaint_chunk_save_can_clear_custom_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "120",
                    "start_seconds": "0.000000",
                    "end_seconds": "5.000000",
                    "custom_seconds": "5.000",
                    "seed": "42",
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)

            with mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                app.update_outpaint_chunk(
                    0,
                    seed="43",
                    prompt_suffix="",
                    custom_seconds="5.000",
                    custom_length=False,
                )

            stored = app.read_outpaint_chunk_rows(manifest)[0]
            self.assertEqual(stored["seed"], "43")
            self.assertEqual(stored["custom_seconds"], "")

    def test_outpaint_chunk_save_persists_prompt_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "120",
                    "start_seconds": "0.000000",
                    "end_seconds": "5.000000",
                    "seed": "42",
                    "prompt_suffix": "",
                    "negative_suffix": "",
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)

            with mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                app.update_outpaint_chunk(
                    0,
                    seed="43",
                    prompt_suffix="avoid changing the actor's hands",
                    negative_suffix="extra fingers",
                )

            stored = app.read_outpaint_chunk_rows(manifest)[0]
            self.assertEqual(stored["prompt_suffix"], "avoid changing the actor's hands")
            self.assertEqual(stored["negative_suffix"], "extra fingers")

    def test_outpaint_chunk_save_persists_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "chunks.csv"
            rows = [{"chunk_index": "0", "start_frame": "0", "end_frame": "120", "seed": "42"}]
            app.write_outpaint_chunk_rows(manifest, rows)

            with mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                app.update_outpaint_chunk(0, seed="42", prompt_suffix="", offset_x="-11", offset_y="7")

            stored = app.read_outpaint_chunk_rows(manifest)[0]
            self.assertEqual(stored["offset_x"], "-11")
            self.assertEqual(stored["offset_y"], "7")

    def test_outpaint_chunk_preview_passes_chunk_offsets(self) -> None:
        settings = {"outpaint": {"target_aspect": "16:9"}}
        fake_state = {
            "rows": [
                {
                    "index": 0,
                    "start": 1.0,
                    "end": 2.0,
                    "fps": 24.0,
                    "offset_x": "13",
                    "offset_y": "-5",
                    "raw_path": "",
                }
            ]
        }
        with (
            mock.patch.object(server, "outpaint_chunks_state", return_value=fake_state),
            mock.patch.object(server, "pipeline_source_text", return_value="input/example.mp4"),
            mock.patch.object(server, "aspect_preview_at", return_value="preview.jpg") as preview,
        ):
            self.assertEqual(app.outpaint_chunk_preview(settings, 0, "source", "middle"), "preview.jpg")

        preview.assert_called_once_with("input/example.mp4", "16:9", 1.5, 13, -5)


    def test_outpaint_chunk_state_reports_exact_prompts_sent_to_comfy(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video")
            manifest = folder / "chunks.csv"
            app.write_outpaint_chunk_rows(
                manifest,
                [
                    {
                        "chunk_index": "0",
                        "start_frame": "0",
                        "end_frame": "24",
                        "seed": "42",
                        "prompt_suffix": "extend the wallpaper",
                        "negative_suffix": "extra hands",
                    }
                ],
            )
            settings = {
                "global": {"source": app.rel(source), "section_start": "0", "section_end": ""},
                "outpaint": {
                    "target_aspect": "16:9",
                    "target_height": "720",
                    "chunk_seconds": "1",
                    "overlap_frames": "0",
                    "prompt": "outpaint with natural edges",
                    "negative_prompt": "text",
                },
            }
            with (
                mock.patch.object(server, "ensure_source_section_clip"),
                mock.patch.object(server, "resolve_video_source", return_value=source),
                mock.patch.object(server, "video_metrics", return_value={"fps": 24.0, "frames": 24}),
                mock.patch.object(server, "outpaint_chunk_manifest_for", return_value=app.rel(manifest)),
                mock.patch.object(server, "outpaint_chunk_dir_for", return_value=folder),
            ):
                state = app.outpaint_chunks_state(settings)

        row = state["rows"][0]
        self.assertEqual(row["effective_prompt"], "outpaint with natural edges. extend the wallpaper")
        self.assertEqual(row["effective_negative_prompt"], "text. extra hands")


    def test_clearing_outpaint_guide_deletes_cached_chunk_guides(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "demo_chunks.csv"
            guide_dir = app.ROOT / "intermediate" / "outpaint_anchors" / manifest.stem
            guide_dir.mkdir(parents=True, exist_ok=True)
            current = guide_dir / "chunk_0000_guide_qwen.png"
            older = guide_dir / "chunk_0000_middle_qwen.png"
            other = guide_dir / "chunk_0001_guide_qwen.png"
            for path in (current, older, other, current.with_suffix(".png.sig.json")):
                path.write_bytes(b"cached")
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "10",
                    "start_seconds": "0",
                    "end_seconds": "1",
                    "seed": "42",
                    "anchor_image": app.rel(current),
                    "anchor_position": "guide",
                    "anchor_seconds": "0.5",
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)

            try:
                with mock.patch.object(server, "outpaint_chunks_state", return_value={"manifest": str(manifest)}):
                    app.clear_outpaint_anchor(0)

                self.assertFalse(current.exists())
                self.assertFalse(older.exists())
                self.assertFalse(current.with_suffix(".png.sig.json").exists())
                self.assertTrue(other.exists())
                self.assertEqual(app.read_outpaint_chunk_rows(manifest)[0]["anchor_image"], "")
            finally:
                shutil.rmtree(guide_dir, ignore_errors=True)

    def test_reference_manifest_read_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "refs.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                handle.write("# source_video=input/example.mp4\n")
                writer = csv.DictWriter(handle, fieldnames=["enabled", "end", "source_reference", "color_reference", "prompt"])
                writer.writeheader()
                writer.writerow(
                    {
                        "enabled": "true",
                        "end": "00:00:01.000",
                        "source_reference": "bw.png",
                        "color_reference": "color.png",
                        "prompt": "",
                    }
                )

            source, fields, rows = app.read_manifest_details(manifest)

            self.assertEqual(source, "input/example.mp4")
            self.assertIn("color_reference", fields)
            self.assertEqual(rows[0]["source_reference"], "bw.png")

    def test_shot_fade_marker_round_trips_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            manifest = Path(tmp_text) / "refs.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["enabled", "end", "source_reference", "color_reference", "prompt"])
                writer.writeheader()
                writer.writerow({"enabled": "true", "end": "00:00:01.000", "source_reference": "a.png", "color_reference": "a_color.png", "prompt": ""})
                writer.writerow({"enabled": "true", "end": "00:00:02.000", "source_reference": "b.png", "color_reference": "b_color.png", "prompt": ""})

            app.update_shot_fade(str(manifest), 0, True, "0.5")
            _source, fields, rows = app.read_manifest_details(manifest)

            self.assertIn("fade_to_next", fields)
            self.assertEqual(rows[0]["fade_to_next"], "true")
            self.assertEqual(rows[0]["crossfade_seconds"], "0.5")

    def test_colorize_plan_extends_fading_transition_chunks(self) -> None:
        rows = [
            {"end": "00:00:01.000", "fade_to_next": "true", "crossfade_seconds": "0.5"},
            {"end": "00:00:02.000"},
        ]

        plan, transitions = colorize_video.shot_plan(rows, total_frames=48, fps=24.0)

        self.assertEqual(transitions[0], 12)
        self.assertEqual(plan[0]["start"], 0)
        self.assertEqual(plan[0]["end"], 30)
        self.assertEqual(plan[1]["start"], 18)
        self.assertEqual(plan[1]["end"], 48)

    def test_outpaint_overlap_context_stops_before_guide_inside_overlap(self) -> None:
        self.assertEqual(outpaint_video.overlap_context_before_anchor(8, "0.125", 24.0, 100), 3)
        self.assertEqual(outpaint_video.overlap_context_before_anchor(8, "1.0", 24.0, 100), 8)
        self.assertEqual(outpaint_video.overlap_context_before_anchor(8, "0", 24.0, 100), 0)

    def test_outpaint_overlap_context_rejects_mixed_geometry(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            chunk = folder / "chunk.mp4"
            previous = folder / "previous.mp4"

            with (
                mock.patch.object(outpaint_video, "probe_video") as probe,
                mock.patch.object(outpaint_video.subprocess, "run") as run,
                mock.patch.object(outpaint_video, "replace_with_retry") as replace,
            ):
                probe.side_effect = [
                    {"width": 1280, "height": 720, "frames": 8},
                    {"width": 1280, "height": 704, "frames": 8},
                ]
                previous.write_bytes(b"placeholder")

                with self.assertRaisesRegex(RuntimeError, "working canvas should stay"):
                    outpaint_video.inject_overlap_context("ffmpeg", chunk, previous, 3, 24.0, False)

            run.assert_not_called()
            replace.assert_not_called()

    def test_outpaint_overlap_context_skips_first_new_frame_on_matching_geometry(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            chunk = folder / "chunk.mp4"
            previous = folder / "previous.mp4"

            with (
                mock.patch.object(outpaint_video, "probe_video") as probe,
                mock.patch.object(outpaint_video.subprocess, "run") as run,
                mock.patch.object(outpaint_video, "replace_with_retry") as replace,
            ):
                probe.side_effect = [
                    {"width": 1280, "height": 704, "frames": 8},
                    {"width": 1280, "height": 704, "frames": 8},
                ]
                previous.write_bytes(b"placeholder")

                result = outpaint_video.inject_overlap_context("ffmpeg", chunk, previous, 3, 24.0, False)

            self.assertNotEqual(result, chunk)
            filter_text = run.call_args.args[0][7]
            self.assertNotIn("scale=1280:720", filter_text)
            self.assertIn("trim=start_frame=4:end_frame=8", filter_text)
            replace.assert_called_once()

    def test_command_construction_for_outpaint_uses_overview_source(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update(
            {
                "target_aspect": "16:9",
                "target_height": "720",
                "chunk_seconds": "20",
                "overlap_frames": "8",
                "crop_left": "0",
                "crop_right": "0",
                "crop_top": "0",
                "crop_bottom": "0",
            }
        )

        command = app.APP.command_for("outpaint")

        self.assertIn("--source", command)
        self.assertIn("input/example.mp4", command)
        self.assertIn("--chunk-manifest", command)

    def test_outpaint_stage_defaults_to_longer_chunks(self) -> None:
        outpaint_stage = next(stage for stage in app.STAGES if stage.key == "outpaint")
        chunk_field = next(field for field in outpaint_stage.fields if field[0] == "chunk_seconds")

        self.assertEqual(chunk_field[3], "20")

    def test_comfy_node_check_reports_missing_custom_nodes(self) -> None:
        with mock.patch.object(comfy_api, "object_info", return_value={"UnetLoaderGGUF": {}}):
            with self.assertRaisesRegex(RuntimeError, "ComfyUI-LTXVideo"):
                comfy_api.ensure_node_types(
                    "http://127.0.0.1:8188",
                    {"LTXVImgToVideoConditionOnly": "ComfyUI-LTXVideo", "UnetLoaderGGUF": "ComfyUI-GGUF"},
                    "outpainting workflow",
                )

    def test_mmaudio_graph_selects_distinct_model_files(self) -> None:
        model_files = [
            "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
            "mmaudio_large_44k_v2_fp16.safetensors",
            "mmaudio_synchformer_fp16.safetensors",
            "mmaudio_vae_44k_fp16.safetensors",
        ]
        info = {
            "MMAudioModelLoader": {
                "input": {
                    "required": {
                        "mmaudio_model": (model_files, {}),
                        "base_precision": (["fp16", "fp32"], {"default": "fp16"}),
                    },
                },
            },
            "MMAudioFeatureUtilsLoader": {
                "input": {
                    "required": {
                        "vae_model": (model_files, {}),
                        "synchformer_model": (model_files, {}),
                        "clip_model": (model_files, {}),
                    },
                    "optional": {
                        "mode": (["16k", "44k"], {"default": "44k"}),
                        "precision": (["fp16", "fp32"], {"default": "fp16"}),
                    },
                },
            },
            "MMAudioSampler": {"input": {"required": {"images": ("IMAGE",)}}},
        }

        graph = audio_models.sfx_prompt_graph(
            info,
            video_name="proxy.mp4",
            prompt="machinery",
            negative="music",
            seconds=8,
            steps=25,
            cfg=4.5,
            seed=42,
            prefix="arp_audio_sfx/test",
        )

        self.assertEqual(graph["1"]["inputs"]["mmaudio_model"], "mmaudio_large_44k_v2_fp16.safetensors")
        self.assertEqual(graph["2"]["inputs"]["vae_model"], "mmaudio_vae_44k_fp16.safetensors")
        self.assertEqual(graph["2"]["inputs"]["synchformer_model"], "mmaudio_synchformer_fp16.safetensors")
        self.assertEqual(graph["2"]["inputs"]["clip_model"], "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors")

    def test_music_checkpoint_validation_reports_available_choices(self) -> None:
        info = {
            "CheckpointLoaderSimple": {
                "input": {
                    "required": {
                        "ckpt_name": (["ltx-2.3-22b-dev-fp8.safetensors"], {}),
                    },
                },
            },
        }

        with self.assertRaisesRegex(RuntimeError, "stable_audio_open_1.0.safetensors"):
            audio_models.ensure_checkpoint_choice(info, "stable_audio_open_1.0.safetensors")

    def test_stable_audio_music_graph_loads_separate_t5_encoder(self) -> None:
        info = {
            "KSampler": {
                "input": {
                    "required": {
                        "sampler_name": (["euler"], {}),
                        "scheduler": (["simple"], {}),
                    },
                },
            },
            "EmptyLatentAudio": {"input": {"required": {"seconds": ("FLOAT", {"default": 30.0}), "batch_size": ("INT", {"default": 1})}}},
        }

        graph = audio_models.music_prompt_graph(
            info,
            checkpoint="stable_audio_open_1.0.safetensors",
            text_encoder="t5_base.safetensors",
            prompt="orchestral",
            negative="noise",
            seconds=12.0,
            steps=20,
            cfg=6.0,
            seed=42,
            prefix="arp_audio/music_test",
        )

        self.assertEqual(graph["2"]["class_type"], "CLIPLoader")
        self.assertEqual(graph["2"]["inputs"]["clip_name"], "t5_base.safetensors")
        self.assertEqual(graph["2"]["inputs"]["type"], "stable_audio")
        self.assertEqual(graph["5"]["class_type"], "ConditioningStableAudio")
        self.assertEqual(graph["7"]["inputs"]["positive"], ["5", 0])
        self.assertEqual(graph["7"]["inputs"]["negative"], ["5", 1])
        self.assertEqual(graph["8"]["inputs"]["samples"], ["7", 0])

    def test_music_checkpoint_file_preflight_reports_gated_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            with self.assertRaisesRegex(FileNotFoundError, "stabilityai/stable-audio-open-1.0"):
                create_audio_track.ensure_music_checkpoint_file(Path(tmp_text), "stable_audio_open_1.0.safetensors")

    def test_audio_music_missing_checkpoint_opens_huggingface_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            comfy = Path(tmp_text) / "ComfyUI"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update(
                {
                    "source": str(source),
                    "expand_outpaint": "false",
                    "colorize": "false",
                    "add_soundtrack": "true",
                    "section_start": "0",
                    "section_end": "",
                }
            )
            app.APP.settings["audio"].update(
                {
                    "create_music": "true",
                    "create_sfx": "false",
                    "music_checkpoint": "stable_audio_open_1.0.safetensors",
                }
            )

            with (
                mock.patch.object(server, "current_config", return_value={"comfy_dir": str(comfy)}),
                mock.patch.object(server, "ROOT", Path(tmp_text)),
                mock.patch.object(server.webbrowser, "open", return_value=True) as open_browser,
                mock.patch.object(server, "ensure_comfy_available_for_stage") as ensure_comfy,
            ):
                ok, message = app.APP.run_stage("audio")

        self.assertFalse(ok)
        self.assertIn("Hugging Face license acceptance", message)
        open_browser.assert_called_once_with(server.STABLE_AUDIO_LICENSE_URL)
        ensure_comfy.assert_not_called()

    def test_audio_music_second_click_after_handoff_attempts_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            source = root / "source.mp4"
            comfy = root / "ComfyUI"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update(
                {
                    "source": str(source),
                    "expand_outpaint": "false",
                    "colorize": "false",
                    "add_soundtrack": "true",
                    "section_start": "0",
                    "section_end": "",
                }
            )
            app.APP.settings["audio"].update(
                {
                    "create_music": "true",
                    "create_sfx": "false",
                    "music_checkpoint": "stable_audio_open_1.0.safetensors",
                }
            )

            with (
                mock.patch.object(server, "current_config", return_value={"comfy_dir": str(comfy)}),
                mock.patch.object(server, "ROOT", root),
                mock.patch.object(server.webbrowser, "open", return_value=True) as open_browser,
            ):
                marker = server.stable_audio_handoff_marker_path("stable_audio_open_1.0.safetensors")
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}", encoding="utf-8")
                ok, message = server.stable_audio_browser_handoff("stable_audio_open_1.0.safetensors")

        self.assertTrue(ok)
        self.assertEqual(message, "")
        open_browser.assert_not_called()

    def test_audio_state_exposes_lossless_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            root = Path(tmp_text)
            source = root / "input" / "Silent Movie.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"not a real video")
            stem_dir = root / ".cache" / "audio" / "Silent_Movie"
            stem_dir.mkdir(parents=True)
            (stem_dir / "music_stem.wav").write_bytes(b"music")
            (stem_dir / "sfx_stem.wav").write_bytes(b"sfx")

            app.APP.settings["global"].update(
                {
                    "source": str(source),
                    "expand_outpaint": "false",
                    "colorize": "false",
                    "add_soundtrack": "true",
                    "section_start": "0",
                    "section_end": "",
                }
            )
            with mock.patch.object(server, "ROOT", root):
                payload = app.APP.state("audio")

            stems = {row["key"]: row for row in payload["audio_stems"]}
            self.assertEqual(set(stems), {"music", "sfx", "mixed"})
            self.assertTrue(stems["music"]["exists"])
            self.assertTrue(stems["sfx"]["exists"])
            self.assertFalse(stems["mixed"]["exists"])
            self.assertEqual(stems["music"]["path"], ".cache/audio/Silent_Movie/music_stem.wav")
            self.assertEqual(stems["sfx"]["path"], ".cache/audio/Silent_Movie/sfx_stem.wav")

    def test_source_section_names_include_trim_points(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        first = app.source_section_output_for(app.APP.settings)
        app.APP.settings["global"].update({"section_start": "45", "section_end": "60"})
        second = app.source_section_output_for(app.APP.settings)

        self.assertNotEqual(first, second)
        self.assertIn("0000012000_0000024000", first.name)
        self.assertIn("0000045000_0000060000", second.name)

    def test_pipeline_source_uses_section_when_trim_points_are_set(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        self.assertIn("source_sections", app.pipeline_source_text(app.APP.settings))

    def test_opening_source_resets_trim_to_source_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "new-source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": "input/old.mp4", "section_start": "10", "section_end": "20"})

            with (
                mock.patch.object(server, "video_metrics", return_value={"duration": 123.456}),
                mock.patch.object(server, "ffprobe_basic_info", return_value={"resolution": "1920x1080"}),
            ):
                app.APP.update_settings("global", {"source": str(source)})

        self.assertEqual(app.APP.settings["global"]["section_start"], "0")
        self.assertEqual(app.APP.settings["global"]["section_end"], "123.456")

    def test_opening_squareish_sd_source_enables_outpaint_and_upscale_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "new-source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": "input/old.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "false"})

            with (
                mock.patch.object(server, "video_metrics", return_value={"duration": 60.0}),
                mock.patch.object(server, "ffprobe_basic_info", return_value={"resolution": "960x720"}),
            ):
                app.APP.update_settings("global", {"source": str(source)})

        self.assertEqual(app.APP.settings["global"]["expand_outpaint"], "true")
        self.assertEqual(app.APP.settings["global"]["upscale"], "true")
        self.assertEqual(app.APP.settings["outpaint"]["target_aspect"], "16:9")
        self.assertEqual(app.APP.settings["outpaint"]["target_height"], "source")
        self.assertEqual(app.APP.settings["outpaint"]["seed_qwen_guides"], "false")
        self.assertEqual(app.APP.settings["upscale"]["target_width"], "1920")
        self.assertEqual(app.APP.settings["upscale"]["target_height"], "1080")

    def test_opening_1080p_widescreen_source_disables_outpaint_and_upscale_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "new-source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": "input/old.mp4", "expand_outpaint": "true", "colorize": "true", "upscale": "true"})

            with (
                mock.patch.object(server, "video_metrics", return_value={"duration": 60.0}),
                mock.patch.object(server, "ffprobe_basic_info", return_value={"resolution": "1920x1080"}),
            ):
                app.APP.update_settings("global", {"source": str(source)})

        self.assertEqual(app.APP.settings["global"]["expand_outpaint"], "false")
        self.assertEqual(app.APP.settings["global"]["upscale"], "false")
        self.assertEqual(app.APP.settings["outpaint"]["target_height"], "720")
        self.assertEqual(app.APP.settings["outpaint"]["seed_qwen_guides"], "false")

    def test_load_settings_never_persists_qwen_seed_guides_as_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            settings_file = Path(tmp_text) / "settings.json"
            settings_file.write_text(json.dumps({"outpaint": {"seed_qwen_guides": "true"}}), encoding="utf-8")
            with mock.patch.object(app, "SETTINGS_FILE", settings_file), mock.patch.object(app, "newest", return_value=None):
                settings = app.load_settings()

        self.assertEqual(settings["outpaint"]["seed_qwen_guides"], "false")

    def test_detected_monochrome_source_sets_colorize_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            source = Path(tmp_text) / "new-source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": str(source), "colorize": "false"})

            app.APP.apply_detected_source_tone(str(source), True)
            self.assertEqual(app.APP.settings["global"]["colorize"], "true")

            app.APP.apply_detected_source_tone(str(source), False)
            self.assertEqual(app.APP.settings["global"]["colorize"], "false")

    def test_project_payload_round_trips_settings_with_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "demo.arpp"
            app.APP.settings["global"].update({"source": "input/example.mp4"})
            path.write_text(json.dumps(app.project_payload(app.APP.settings)), encoding="utf-8")

            loaded = app.read_project_file(path)

        self.assertEqual(loaded["global"]["source"], "input/example.mp4")
        self.assertIn("schema_version", app.project_payload(app.APP.settings))

    def test_project_bundle_includes_reference_assets_without_openai_key(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            project = folder / "demo.arpp"
            source.write_bytes(b"bw")
            color.write_bytes(b"color")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color)}],
            )
            settings = copy.deepcopy(app.APP.settings)
            settings["references"].update({"manifest": app.rel(manifest), "openai_api_key": "sk-secret"})

            app.write_project_file(project, settings)

            with zipfile.ZipFile(project) as archive:
                names = set(archive.namelist())
                payload = json.loads(archive.read("project.json").decode("utf-8"))

            self.assertIn(app.rel(manifest).replace("\\", "/"), names)
            self.assertIn(app.rel(source).replace("\\", "/"), names)
            self.assertIn(app.rel(color).replace("\\", "/"), names)
            self.assertNotIn("openai_api_key", payload["settings"]["references"])

    def test_openai_reference_command_requires_key_and_uses_selected_model(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            source.write_bytes(b"bw")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference", "prompt"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color), "prompt": "warm highlights"}],
            )
            app.APP.settings["references"].update({"openai_api_key": "", "openai_image_model": "gpt-image-2"})

            with self.assertRaisesRegex(RuntimeError, "OpenAI API key"):
                app.openai_reference_regeneration_command(str(manifest), 0)

            app.APP.settings["references"].update({"openai_api_key": "sk-test", "openai_image_model": "gpt-image-1"})
            command, output = app.openai_reference_regeneration_command(str(manifest), 0)

            self.assertIn("openai_generate_reference.py", " ".join(command))
            self.assertIn("--manifest", command)
            self.assertIn("--row-index", command)
            self.assertEqual(command[command.index("--model") + 1], "gpt-image-1")
            self.assertEqual(output, app.rel(color))

    def test_openai_manifest_command_can_send_nearby_reference_images(self) -> None:
        app.APP.settings["references"].update(
            {
                "manifest": "manifests/references/demo.csv",
                "method": "openai",
                "openai_api_key": "sk-test",
                "openai_image_model": "gpt-image-2",
                "openai_send_references": "true",
            }
        )

        command = app.APP.command_for("references")

        self.assertIn("openai_generate_reference.py", " ".join(command))
        self.assertIn("--reference-count", command)
        self.assertNotIn("qwen_colorize_references.py", " ".join(command))

    def test_openai_nearby_references_prefer_previous_then_later(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            refs = []
            rows = []
            for index in range(5):
                color = folder / f"color_{index}.png"
                if index in {0, 2, 3, 4}:
                    color.write_bytes(b"ref")
                refs.append(color)
                rows.append({"color_reference": app.rel(color)})

            chosen = openai_generate_reference.nearby_reference_images(rows, 1, 3)
            early = openai_generate_reference.nearby_reference_images(rows, 0, 3)

        self.assertEqual(chosen, [refs[0], refs[2], refs[3]])
        self.assertEqual(early, [refs[2], refs[3], refs[4]])

    def test_reference_edit_recent_references_prefer_previous_then_later(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            refs = []
            rows = []
            for index in range(5):
                color = folder / f"color_{index}.png"
                if index in {0, 2, 3, 4}:
                    color.write_bytes(b"ref")
                refs.append(app.rel(color))
                rows.append({"color_reference": app.rel(color)})

            chosen = app.recent_color_references(rows, 1, 3)
            early = app.recent_color_references(rows, 0, 3)

        self.assertEqual(chosen, [refs[0], refs[2], refs[3]])
        self.assertEqual(early, [refs[2], refs[3], refs[4]])

    def test_sam2_mask_helper_uses_point_prompts_and_returns_mask_data_url(self) -> None:
        import numpy as np
        from PIL import Image

        class FakePredictor:
            def __init__(self) -> None:
                self.image_shape = None
                self.point_coords = None
                self.point_labels = None

            def set_image(self, image) -> None:
                self.image_shape = image.shape

            def predict(self, point_coords=None, point_labels=None, multimask_output=True):
                self.point_coords = point_coords
                self.point_labels = point_labels
                masks = np.zeros((2, 4, 4), dtype=np.uint8)
                masks[1, 1:3, 1:3] = 1
                return masks, np.array([0.2, 0.9]), None

        predictor = FakePredictor()
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            image = Path(tmp_text) / "source.png"
            Image.new("RGB", (4, 4), (20, 30, 40)).save(image)
            with mock.patch.object(sam_masks, "_sam2_predictor", return_value=predictor):
                result = sam_masks.sam2_mask_for_image(
                    image,
                    [{"x": 1, "y": 1, "label": "add"}, {"x": 2, "y": 2, "label": "subtract"}],
                    4,
                    4,
                )

        self.assertTrue(result["mask"].startswith("data:image/png;base64,"))
        self.assertIn("SAM 2.1 Hiera Large", result["provider"])
        self.assertEqual(predictor.image_shape, (4, 4, 3))
        self.assertEqual(predictor.point_labels.tolist(), [1, 0])
        self.assertEqual(predictor.point_coords.tolist(), [[1.0, 1.0], [2.0, 2.0]])

    def test_reference_edit_preview_command_selects_masked_and_unmasked_runners(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            masked_workflow = folder / "masked.json"
            source.write_bytes(b"bw")
            color.write_bytes(b"color")
            masked_workflow.write_text("{}", encoding="utf-8")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference", "prompt"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color), "prompt": ""}],
            )
            app.APP.settings["references"].update(
                {
                    "workflow": "workflows/qwen_image_edit/Image Edit (Qwen 2511).json",
                    "masked_workflow": app.rel(masked_workflow),
                }
            )
            unmasked, unmasked_output = app.reference_edit_preview_command(app.rel(manifest), 0, "make it green")
            masked, masked_output = app.reference_edit_preview_command(app.rel(manifest), 0, "make it green", "iVBORw0KGgo=")

        self.assertIn("generate_single_reference.py", " ".join(unmasked))
        self.assertIn("edit_reference_image.py", " ".join(masked))
        self.assertIn("--mask", masked)
        self.assertIn("outpainted_references_color_edits", unmasked_output)
        self.assertIn("outpainted_references_color_edits", masked_output)

    def test_reference_edit_accept_and_revert_updates_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            edit = folder / "edit.png"
            source.write_bytes(b"bw")
            color.write_bytes(b"color")
            edit.write_bytes(b"edit")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference", "prompt"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color), "prompt": ""}],
            )

            accepted = app.accept_reference_edit(app.rel(manifest), 0, app.rel(edit))
            after_accept = app.read_manifest(manifest)[0]
            reverted = app.revert_reference_edit(app.rel(manifest), 0)
            after_revert = app.read_manifest(manifest)[0]

        self.assertEqual(accepted["color_reference"], app.rel(edit))
        self.assertEqual(after_accept["color_reference_previous"], app.rel(color))
        self.assertEqual(reverted["color_reference"], app.rel(color))
        self.assertEqual(after_revert["color_reference_previous"], app.rel(edit))

    def test_guide_edit_preview_command_always_uses_masked_runner(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            guide = folder / "guide.png"
            prepared = folder / "prepared.mp4"
            prepared_preview = folder / "prepared_preview.png"
            masked_workflow = folder / "masked.json"
            Image.new("RGB", (16, 9), (32, 32, 32)).save(guide)
            Image.new("RGB", (16, 9), (0, 0, 0)).save(prepared_preview)
            prepared.write_bytes(b"prepared")
            masked_workflow.write_text("{}", encoding="utf-8")
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "24",
                    "start_seconds": "0",
                    "end_seconds": "1",
                    "guide_frames": json.dumps([{"frame_idx": 0, "strength": 0.7, "image": app.rel(guide), "seed": True}]),
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)
            app.APP.settings["references"].update(
                {
                    "workflow": "workflows/qwen_image_edit/Image Edit (Qwen 2511).json",
                    "masked_workflow": app.rel(masked_workflow),
                }
            )
            fake_state = {"manifest": app.rel(manifest), "rows": [{"index": 0}]}
            with (
                mock.patch.object(outpaint_guides, "outpaint_chunks_state", return_value=fake_state),
                mock.patch.object(outpaint_guides, "ensure_outpaint_prepared_canvas", return_value=prepared),
                mock.patch.object(outpaint_guides, "chunk_frame_preview", return_value=app.rel(prepared_preview)),
            ):
                unmasked, unmasked_output = app.guide_edit_preview_command(0, 0, "replace detail")
                defaulted, _defaulted_output = app.guide_edit_preview_command(0, 0, "")
                masked, masked_output = app.guide_edit_preview_command(0, 0, "replace detail", "iVBORw0KGgo=")

        self.assertIn("edit_reference_image.py", " ".join(unmasked))
        self.assertIn("edit_reference_image.py", " ".join(masked))
        self.assertEqual(Path(unmasked[unmasked.index("--source-image") + 1]), app.resolve(app.rel(guide)))
        self.assertEqual(defaulted[defaulted.index("--instruction") + 1], "Replace the black bars.")
        self.assertIn("--mask", unmasked)
        self.assertIn("--mask", masked)
        self.assertIn("outpaint_guides", unmasked_output)
        self.assertIn("outpaint_guides", masked_output)

    def test_outpaint_guide_generation_defaults_to_replace_black_bars(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            prepared = folder / "prepared.mp4"
            source_frame = folder / "source.jpg"
            masked_workflow = folder / "masked.json"
            prepared.write_bytes(b"prepared")
            from PIL import Image

            Image.new("RGB", (16, 9), (0, 0, 0)).save(source_frame)
            masked_workflow.write_text("{}", encoding="utf-8")
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "24",
                    "start_seconds": "0",
                    "end_seconds": "1",
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)
            app.APP.settings["references"].update({"masked_workflow": app.rel(masked_workflow)})
            fake_state = {"manifest": app.rel(manifest), "rows": [{"index": 0, "fps": 24, "start": 0.0, "end": 1.0}]}

            with (
                mock.patch.object(outpaint_guides, "outpaint_chunks_state", return_value=fake_state),
                mock.patch.object(outpaint_guides, "pipeline_source_text", return_value="input/example.mp4"),
                mock.patch.object(outpaint_guides, "ensure_outpaint_prepared_canvas", return_value=prepared),
                mock.patch.object(outpaint_guides, "chunk_frame_preview", return_value=app.rel(source_frame)),
            ):
                command, _output, _canvas, _seconds = app.outpaint_guide_generation_command(0, "")

        self.assertIn("edit_reference_image.py", " ".join(command))
        self.assertEqual(command[command.index("--instruction") + 1], "Replace the black bars.")
        self.assertIn("--mask", command)

    def test_auto_edge_mask_uses_sloppy_feathered_boundary(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "pillarbox.png"
            mask_path = folder / "mask.png"
            image = Image.new("RGB", (96, 48), (0, 0, 0))
            for y in range(48):
                for x in range(24, 72):
                    image.putpixel((x, y), (64, 64, 64))
            image.save(source)

            guide_frame_utils.save_edge_mask_for_image(source, mask_path)

            with Image.open(mask_path) as mask:
                pixels = mask.convert("L").load()
                left_edges = []
                for y in range(mask.height):
                    xs = [x for x in range(mask.width // 2) if pixels[x, y] > 0]
                    left_edges.append(max(xs))

                self.assertEqual(pixels[0, mask.height // 2], 255)
                self.assertEqual(pixels[mask.width // 2, mask.height // 2], 0)
                self.assertGreater(len(set(left_edges)), 4)
                self.assertTrue(any(0 < pixels[x, mask.height // 2] < 255 for x in range(mask.width // 2)))

    def test_masked_edit_uses_bundled_workflow_when_setting_is_empty(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "refs.csv"
            source = folder / "bw.png"
            color = folder / "color.png"
            source.write_bytes(b"bw")
            color.write_bytes(b"color")
            app.write_manifest_details(
                manifest,
                "input/example.mp4",
                ["enabled", "end", "source_reference", "color_reference", "prompt"],
                [{"enabled": "true", "end": "00:00:01.000", "source_reference": app.rel(source), "color_reference": app.rel(color), "prompt": ""}],
            )
            app.APP.settings["references"].update({"masked_workflow": ""})

            command, _output = app.reference_edit_preview_command(app.rel(manifest), 0, "make it green", "iVBORw0KGgo=")

        self.assertIn("edit_reference_image.py", " ".join(command))
        self.assertIn("workflows/qwen_image_edit/Image Edit Inpaint (Qwen 2511).json", command)

    def test_masked_edit_workflow_patches_user_instruction_into_positive_prompt(self) -> None:
        import argparse

        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            comfy_dir = folder / "comfy"
            source = folder / "source.png"
            mask = folder / "mask.png"
            source.write_bytes(b"source")
            mask.write_bytes(b"mask")
            workflow = qwen_colorize_references.load_workflow(app.ROOT / "workflows" / "qwen_image_edit" / "Image Edit Inpaint (Qwen 2511).json")
            args = argparse.Namespace(
                comfy_dir=comfy_dir,
                load_image_node_id="auto",
                mask_image_node_id="auto",
                save_node_id="auto",
                prompt_node_id=None,
                load_image_widget="0",
                mask_image_widget="0",
                prompt_widget="0",
                save_prefix_widget="0",
            )

            prompt = edit_reference_image.patch_masked_workflow(
                args,
                workflow,
                source,
                mask,
                folder / "output.png",
                "Replace the masked hat with a bright green hat.",
            )

        self.assertEqual(prompt["1"]["inputs"]["prompt"], "Replace the masked hat with a bright green hat.")
        self.assertEqual(prompt["12"]["inputs"]["image"], "arp_qwen_ref_masks/mask.png")
        self.assertEqual(prompt["11"]["inputs"]["image"], "arp_qwen_ref_edits/source.png")

    def test_guide_edit_accept_and_revert_updates_guide_frames(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "chunks.csv"
            guide = folder / "guide.png"
            edit = folder / "edit.png"
            guide.write_bytes(b"guide")
            edit.write_bytes(b"edit")
            rows = [
                {
                    "chunk_index": "0",
                    "start_frame": "0",
                    "end_frame": "24",
                    "start_seconds": "0",
                    "end_seconds": "1",
                    "guide_frames": json.dumps([{"frame_idx": 0, "strength": 0.7, "image": app.rel(guide)}]),
                }
            ]
            app.write_outpaint_chunk_rows(manifest, rows)
            fake_state = {"manifest": app.rel(manifest), "rows": [{"index": 0}]}
            with mock.patch.object(outpaint_guides, "outpaint_chunks_state", return_value=fake_state):
                accepted = app.accept_guide_edit(0, 0, app.rel(edit))
                accepted_frame = outpaint_guides._parse_guide_frames(app.read_outpaint_chunk_rows(manifest)[0])[0]
                reverted = app.revert_guide_edit(0, 0)
                reverted_frame = outpaint_guides._parse_guide_frames(app.read_outpaint_chunk_rows(manifest)[0])[0]

        self.assertEqual(accepted["image"], app.rel(edit))
        self.assertEqual(accepted_frame["image_previous"], app.rel(guide))
        self.assertNotIn("seed", accepted_frame)
        self.assertEqual(reverted["image"], app.rel(guide))
        self.assertEqual(reverted_frame["image_previous"], app.rel(edit))

    def test_project_save_suggestion_uses_last_browse_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            app.APP.settings["global"].update({"source": "input/example.mp4", "last_browse_dir": str(folder)})

            suggestion = app.project_save_suggestion(app.APP.settings)

        self.assertEqual(suggestion.parent, folder)
        self.assertEqual(suggestion.name, "example.arpp")

    def test_browse_initial_path_uses_last_browse_dir_without_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            folder = Path(tmp_text)
            app.APP.settings["global"]["last_browse_dir"] = str(folder)

            self.assertEqual(app.browse_initial_path("project_open", ""), folder)

    def test_browse_initial_path_prefers_last_dir_over_existing_current_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text, tempfile.TemporaryDirectory() as old_text:
            remembered = Path(tmp_text)
            old = Path(old_text)
            current = old / "layer.mp4"
            current.write_bytes(b"placeholder")
            app.APP.settings["global"]["last_browse_dir"] = str(remembered)

            self.assertEqual(app.browse_initial_path("save", str(current)), remembered / "layer.mp4")
            self.assertEqual(app.browse_initial_path("file", str(current)), remembered)

    def test_colorized_outputs_include_both_methods(self) -> None:
        outputs = app.colorized_outputs_for_manifest("manifests/references/colorize_manifest_demo_shots_auto.csv", "both")

        self.assertEqual(len(outputs), 2)
        # deepexemplar and colormnet are distinct identity-keyed artifacts.
        self.assertNotEqual(outputs[0], outputs[1])
        for output in outputs:
            self.assertTrue(output.startswith("intermediate/outpainted_colorized/"), output)
            self.assertTrue(output.endswith(".mp4"))

    def test_colorization_command_can_request_both_methods(self) -> None:
        app.APP.settings["colour"].update({"manifest": "manifests/references/colorize_manifest_demo_shots_auto.csv", "method": "both"})

        command = app.APP.command_for("colour")

        self.assertIn("--method", command)
        self.assertIn("both", command)
        self.assertNotIn("--output", command)

    def test_skip_outpainting_uses_pipeline_source_for_shot_detection(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "true", "section_start": "0", "section_end": ""})

        app.APP.hydrate_stage_inputs("global")
        stage_keys = [stage.key for stage in app.APP.active_stages()]
        command = app.APP.command_for("shots")

        self.assertNotIn("outpaint", stage_keys)
        self.assertIn("shots", stage_keys)
        self.assertEqual(app.APP.settings["shots"]["outpainted_video"], "input/example.mp4")
        self.assertIn("input/example.mp4", command)

    def test_no_overview_steps_selected_leaves_only_output_tab(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "false", "section_start": "0", "section_end": ""})

        self.assertEqual(app.APP.active_stages(), ())
        self.assertEqual(app.APP.phase_progress()["stages"], [])

    def test_optional_phase_combinations_have_expected_stage_order(self) -> None:
        cases = [
            (False, False, False, []),
            (True, False, False, ["outpaint", "recomp"]),
            (False, True, False, ["shots", "references", "colour", "recomp"]),
            (False, False, True, ["upscale"]),
            (True, True, False, ["outpaint", "shots", "references", "colour", "recomp"]),
            (True, False, True, ["outpaint", "recomp", "upscale"]),
            (False, True, True, ["shots", "references", "colour", "recomp", "upscale"]),
            (True, True, True, ["outpaint", "shots", "references", "colour", "recomp", "upscale"]),
        ]
        for outpaint, colorize, upscale, expected in cases:
            with self.subTest(outpaint=outpaint, colorize=colorize, upscale=upscale):
                app.APP.settings["global"].update(
                    {
                        "source": "input/example.mp4",
                        "expand_outpaint": "true" if outpaint else "false",
                        "colorize": "true" if colorize else "false",
                        "upscale": "true" if upscale else "false",
                        "section_start": "0",
                        "section_end": "",
                    }
                )

                self.assertEqual([stage.key for stage in app.APP.active_stages()], expected)

    def test_outpaint_without_colour_can_feed_upscale_after_recomposition(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "true", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update({"target_aspect": "16:9", "target_height": "720", "crop_left": "0", "crop_right": "0", "crop_top": "0", "crop_bottom": "0"})

        outpainted = app.outpaint_output_for("input/example.mp4", "16:9", "720")
        outpainted_path = app.resolve(outpainted)
        outpainted_path.parent.mkdir(parents=True, exist_ok=True)
        outpainted_path.write_bytes(b"placeholder")
        try:
            app.APP.hydrate_stage_inputs("outpaint")
            recomp_output = app.recomposition_output_for(outpainted)
            command = app.APP.command_for("upscale")
        finally:
            outpainted_path.unlink(missing_ok=True)

        self.assertEqual([stage.key for stage in app.APP.active_stages()], ["outpaint", "recomp", "upscale"])
        self.assertEqual(app.APP.settings["upscale"]["input_video"], recomp_output)
        self.assertEqual(command[command.index("--input") + 1], recomp_output)

    def test_upscale_after_earlier_phase_waits_for_recomposition_output(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "true", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["recomp"]["output"] = "output/reassembled/missing_recomposition.mp4"

        ok, message = app.APP.run_stage("upscale")

        self.assertFalse(ok)
        self.assertIn("Run Recomposition first", message)

    def test_outpaint_hydration_does_not_pick_stale_newest_output_for_new_source(self) -> None:
        app.APP.settings["global"].update({"source": "input/new-source.mp4", "expand_outpaint": "true", "colorize": "false", "upscale": "false", "section_start": "0", "section_end": ""})
        stale = app.ROOT / "intermediate" / "outpainted" / "old-source_16x9_1280x704_outpainted.mp4"

        with mock.patch.object(server, "newest", return_value=stale):
            app.APP.hydrate_stage_inputs("upscale")

        self.assertEqual(app.APP.settings["recomp"]["outpainted_video"], "")

    def test_upscale_only_uses_pipeline_source(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update({"target_width": "1920", "target_height": "1080"})

        app.APP.hydrate_stage_inputs("global")
        stage_keys = [stage.key for stage in app.APP.active_stages()]
        command = app.APP.command_for("upscale")

        self.assertEqual(stage_keys, ["upscale"])
        self.assertEqual(app.APP.settings["upscale"]["input_video"], "input/example.mp4")
        self.assertNotIn("--method", command)
        self.assertIn("--target-width", command)
        self.assertIn("1920", command)
        self.assertIn("--comfy-url", command)
        self.assertEqual(command[command.index("--flashvsr-model") + 1], "FlashVSR-v1.1")
        self.assertEqual(command[command.index("--flashvsr-mode") + 1], "tiny")
        self.assertIn("--flashvsr-tiled-vae", command)
        self.assertIn("--flashvsr-tiled-dit", command)
        self.assertNotIn("--flashvsr-unload-dit", command)
        self.assertEqual(command[command.index("--chunk-seconds") + 1], "6")
        self.assertEqual(command[command.index("--overlap-frames") + 1], "8")
        self.assertIn("scripts\\upscale_video.py", " ".join(command))

    def test_upscale_can_request_flashvsr_unload_dit(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update({"target_width": "1920", "target_height": "1080", "flashvsr_unload_dit": "true"})

        command = app.APP.command_for("upscale")

        self.assertIn("--flashvsr-unload-dit", command)

    def test_upscale_preview_clips_directly_from_selected_source_section(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            source.write_bytes(b"placeholder")
            app.APP.settings["global"].update(
                {
                    "source": app.rel(source),
                    "expand_outpaint": "false",
                    "colorize": "false",
                    "upscale": "true",
                    "section_start": "1",
                    "section_end": "2",
                }
            )
            app.APP.settings["upscale"].update({"target_width": "1920", "target_height": "1080"})

            with (
                mock.patch.object(server, "ensure_source_section_clip") as ensure,
                mock.patch.object(server, "media_clip_path", side_effect=RuntimeError("clip probe")) as clip,
            ):
                ok, message = app.APP.run_upscale_preview()

        self.assertFalse(ok)
        ensure.assert_not_called()
        clip.assert_called_once()
        self.assertEqual(clip.call_args.args[0], source)
        self.assertAlmostEqual(clip.call_args.args[1], 1.0)
        self.assertAlmostEqual(clip.call_args.args[2], 2.0)
        self.assertIn("clip probe", message)
        self.assertNotIn("input does not exist", message)

    def test_upscale_preview_completion_does_not_hydrate_shot_detection(self) -> None:
        class FakeProcess:
            stdout = ["Wrote upscaled video: output/upscaled/previews/example.mp4\n"]

            def wait(self) -> int:
                return 0

        original_log = app.APP.log
        app.APP.log = []
        app.APP.running_stage = "Upscale Preview"
        app.APP.running_stage_key = "upscale"

        try:
            with mock.patch.object(app.APP, "hydrate_stage_inputs") as hydrate:
                app.APP._collect_output(FakeProcess(), "upscale_preview")
        finally:
            log = app.APP.log
            app.APP.log = original_log
            app.APP.running_stage = ""
            app.APP.running_stage_key = ""

        self.assertNotIn(mock.call("shots"), hydrate.call_args_list)
        self.assertIn("Upscale preview ready.", log)
        self.assertFalse(any("Updated Shot Detection input" in line for line in log))

    def test_upscale_preview_state_uses_generated_clip_output_pair(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            clip = Path(tmp_text) / "preview-clip.mp4"
            output = app.ROOT / "output" / "upscaled" / "previews" / "preview-clip_flashvsr_1920x1080_preview_6s.mp4"
            source.write_bytes(b"placeholder")
            clip.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": app.rel(source), "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"placeholder")
            app.APP.settings["upscale"].update({"preview_source": app.rel(clip), "preview_output": app.rel(output)})

            try:
                preview = app.APP.upscale_preview_state()
            finally:
                output.unlink(missing_ok=True)

        self.assertEqual(preview["source"], app.rel(clip))
        self.assertEqual(preview["output"], app.rel(output))
        self.assertEqual(preview["exists"], "true")

    def test_upscale_preview_state_promotes_finished_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            preview_clip = Path(tmp_text) / "preview-clip.mp4"
            source.write_bytes(b"placeholder")
            preview_clip.write_bytes(b"placeholder")
            app.APP.settings["global"].update({"source": app.rel(source), "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
            app.APP.settings["upscale"].update(
                {
                    "target_width": "1920",
                    "target_height": "1080",
                    "preview_source": app.rel(preview_clip),
                    "preview_output": "output/upscaled/previews/stale_preview.mp4",
                }
            )
            output = app.resolve(app.upscale_output_for(app.rel(source), app.APP.settings["upscale"]))
            preview_output = app.resolve(app.APP.settings["upscale"]["preview_output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"placeholder")
            preview_output.write_bytes(b"placeholder")

            try:
                preview = app.APP.upscale_preview_state()
            finally:
                output.unlink(missing_ok=True)
                preview_output.unlink(missing_ok=True)

        self.assertEqual(preview["source"], app.rel(source))
        self.assertEqual(preview["output"], app.rel(output))
        self.assertEqual(preview["kind"], "output")
        self.assertEqual(preview["title"], "Upscale Output")

    def test_upscale_preview_start_records_actual_clip_and_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            source = Path(tmp_text) / "source.mp4"
            clip = Path(tmp_text) / "clip.mp4"
            source.write_bytes(b"placeholder")
            clip.write_bytes(b"placeholder")
            app.APP.settings["global"].update(
                {
                    "source": app.rel(source),
                    "expand_outpaint": "false",
                    "colorize": "false",
                    "upscale": "true",
                    "section_start": "0",
                    "section_end": "",
                }
            )
            app.APP.settings["upscale"].update({"target_width": "1920", "target_height": "1080", "preview_seconds": "6"})

            class FakeProcess:
                stdout = []

                def poll(self):
                    return None

            with (
                mock.patch.object(server, "media_clip_path", return_value=clip),
                mock.patch.object(server.subprocess, "Popen", return_value=FakeProcess()),
                mock.patch.object(server.threading.Thread, "start", lambda _self: None),
            ):
                ok, message = app.APP.run_upscale_preview()

        self.assertTrue(ok, message)
        self.assertEqual(app.APP.settings["upscale"]["preview_source"], app.rel(clip))
        self.assertEqual(
            app.APP.settings["upscale"]["preview_output"],
            app.upscale_preview_output_for(app.rel(clip), app.APP.settings["upscale"]),
        )
        self.assertTrue(app.APP.settings["upscale"]["preview_output"].startswith("output/upscaled/previews/clip_upscalepreview_"))

    def test_upscale_only_ignores_stale_recomposition_input(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update(
            {
                "method": "realbasicvsr",
                "input_video": "output/reassembled/old_intermediate_final.mp4",
                "output": "output/upscaled/old_intermediate_final_realbasicvsr_3840x2160.mp4",
                "target_width": "1920",
                "target_height": "1080",
            }
        )

        command = app.APP.command_for("upscale")

        self.assertIn("--input", command)
        self.assertEqual(command[command.index("--input") + 1], "input/example.mp4")
        self.assertEqual(command[command.index("--output") + 1], app.upscale_output_for("input/example.mp4", app.APP.settings["upscale"]))
        self.assertNotIn("--method", command)

    def test_upscale_ignores_stale_backend_method_setting(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "false", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["upscale"].update({"method": "realbasicvsr", "target_width": "1920", "target_height": "1080"})

        command = app.APP.command_for("upscale")

        self.assertNotIn("--method", command)
        self.assertNotIn("--realbasicvsr-repo", command)
        self.assertIn("--comfy-url", command)
        self.assertEqual(command[command.index("--output") + 1], app.upscale_output_for("input/example.mp4", app.APP.settings["upscale"]))

    def test_flashvsr_prompt_uses_video_helper_load_and_combine_nodes(self) -> None:
        args = argparse.Namespace(
            flashvsr_model="FlashVSR-v1.1",
            flashvsr_mode="tiny",
            flashvsr_scale=2,
            flashvsr_tiled_vae=True,
            flashvsr_tiled_dit=True,
            flashvsr_unload_dit=True,
            flashvsr_seed=123,
        )
        info = {
            "FlashVSRNode": {
                "input": {
                    "required": {
                        "frames": ("IMAGE",),
                        "model": (["FlashVSR", "FlashVSR-v1.1"],),
                        "mode": (["tiny", "tiny-long", "full"],),
                        "scale": ("INT", {"default": 4}),
                        "tiled_vae": ("BOOLEAN", {"default": True}),
                        "tiled_dit": ("BOOLEAN", {"default": True}),
                        "unload_dit": ("BOOLEAN", {"default": False}),
                        "seed": ("INT", {"default": 0}),
                    }
                }
            }
        }

        prompt = upscale_video.flashvsr_prompt("example.mp4", 24.0, args, "arp_upscale/example", info)

        self.assertEqual(prompt["1"]["class_type"], "VHS_LoadVideo")
        self.assertEqual(prompt["2"]["class_type"], "FlashVSRNode")
        self.assertEqual(prompt["2"]["inputs"]["frames"], ["1", 0])
        self.assertEqual(prompt["2"]["inputs"]["model"], "FlashVSR-v1.1")
        self.assertEqual(prompt["2"]["inputs"]["mode"], "tiny")
        self.assertEqual(prompt["2"]["inputs"]["scale"], 2)
        self.assertEqual(prompt["2"]["inputs"]["tiled_vae"], True)
        self.assertEqual(prompt["2"]["inputs"]["tiled_dit"], True)
        self.assertEqual(prompt["2"]["inputs"]["unload_dit"], True)
        self.assertEqual(prompt["2"]["inputs"]["seed"], 123)
        self.assertEqual(prompt["3"]["class_type"], "VHS_VideoCombine")
        self.assertEqual(prompt["3"]["inputs"]["images"], ["2", 0])
        self.assertEqual(prompt["3"]["inputs"]["audio"], ["1", 2])

    def test_upscale_chunk_ranges_overlap_without_dropping_frames(self) -> None:
        ranges = upscale_video.chunk_ranges(total_frames=100, fps=10.0, chunk_seconds=3.0, overlap_frames=4)

        self.assertEqual(ranges, [(0, 30, 0), (26, 60, 4), (56, 90, 4), (86, 100, 4)])

    def test_upscale_chunk_ranges_use_single_prompt_when_clip_fits(self) -> None:
        self.assertEqual(upscale_video.chunk_ranges(total_frames=100, fps=25.0, chunk_seconds=6.0, overlap_frames=8), [(0, 0, 100)])

    def test_upscale_runs_after_recomposition_when_processing_is_enabled(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "expand_outpaint": "true", "colorize": "false", "upscale": "true", "section_start": "0", "section_end": ""})
        app.APP.settings["outpaint"].update({"target_aspect": "16:9", "target_height": "720", "crop_left": "0", "crop_right": "0", "crop_top": "0", "crop_bottom": "0"})

        with mock.patch.object(server, "newest", return_value=None):
            app.APP.hydrate_stage_inputs("global")
        stage_keys = [stage.key for stage in app.APP.active_stages()]
        outpainted = app.outpaint_output_for("input/example.mp4", "16:9", "720")
        outpainted_path = app.resolve(outpainted)
        outpainted_path.parent.mkdir(parents=True, exist_ok=True)
        outpainted_path.write_bytes(b"placeholder")
        try:
            app.APP.hydrate_stage_inputs("outpaint")
        finally:
            outpainted_path.unlink(missing_ok=True)

        self.assertEqual(stage_keys, ["outpaint", "recomp", "upscale"])
        self.assertTrue(app.APP.settings["upscale"]["input_video"].startswith("output/reassembled/"))

    def test_new_outpaint_source_does_not_hydrate_empty_manifest_as_repo_root(self) -> None:
        app.APP.settings["global"].update({"source": "input/new-source.mp4", "expand_outpaint": "true", "colorize": "true", "section_start": "0", "section_end": ""})
        app.APP.settings.setdefault("outpaint", {}).update({"target_aspect": "16:9", "target_height": "480"})

        with mock.patch.object(server, "newest", return_value=None):
            app.APP.hydrate_stage_inputs("global")

        self.assertEqual(app.APP.settings["shots"]["outpainted_video"], "")
        self.assertEqual(app.APP.settings["references"]["manifest"], "")
        self.assertEqual(app.APP.settings["colour"]["manifest"], "")
        self.assertEqual(app.APP.settings["recomp"]["manifest"], "")
        self.assertEqual(app.APP.expected_outputs("shots"), [])

    def test_outpaint_progress_surfaces_active_comfy_chunk_globally(self) -> None:
        original_log = app.APP.log
        app.APP.settings["global"].update({"expand_outpaint": "true", "colorize": "false", "upscale": "false"})
        app.APP.running_stage_key = "outpaint"
        app.APP.running_stage = "Outpainting"
        app.APP.run_started_at = time.time() - 120
        app.APP.log = [
            "Outpaint chunk 1/3: frames 0-480",
            "Wrote raw Comfy chunk 1: chunk_0000.mp4",
            "Outpaint chunk 2/3: frames 472-952",
            "Sending prompt nodes: {'5076': 'VHS_VideoCombine'}",
            "Queued ComfyUI prompt: prompt-id",
        ]

        try:
            stage_progress = app.APP.estimate_running_progress()
            global_progress = app.APP.phase_progress()["global"]
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertIn("Chunk 2/3 rendering in ComfyUI (1 done), ETA", stage_progress["label"])
        self.assertIn("Chunk 2/3 rendering in ComfyUI", global_progress["label"])

    def test_upscale_progress_surfaces_active_chunk_with_eta(self) -> None:
        original_log = app.APP.log
        app.APP.settings["global"].update({"expand_outpaint": "false", "colorize": "false", "upscale": "true"})
        app.APP.running_stage_key = "upscale"
        app.APP.running_stage = "Upscaling"
        app.APP.run_started_at = time.time() - 180
        app.APP.log = [
            "Splitting upscaling into 12 chunk(s): 6s chunks, 8 overlap frame(s)",
            "Upscale chunk 1/12: frames 0-144, trim 0",
            "Wrote upscaled chunk: chunk_0001.mp4",
            "Upscale chunk 2/12: frames 136-288, trim 8",
            "Queued ComfyUI prompt: prompt-id",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertIn("Upscale chunk 2/12 rendering in ComfyUI (1 done), ETA", progress["label"])
        self.assertGreater(progress["percent"], 10)
        self.assertLess(progress["percent"], 100)

    def test_upscale_progress_reports_stitching_after_chunks_complete(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "upscale"
        app.APP.running_stage = "Upscaling"
        app.APP.run_started_at = time.time() - 300
        app.APP.log = [
            "Splitting upscaling into 2 chunk(s): 6s chunks, 8 overlap frame(s)",
            "Upscale chunk 1/2: frames 0-144, trim 0",
            "Wrote upscaled chunk: chunk_0001.mp4",
            "Upscale chunk 2/2: frames 136-288, trim 8",
            "Wrote upscaled chunk: chunk_0002.mp4",
            "Stitching upscaled chunks: 2 chunk(s)",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "Upscale chunks complete, stitching")
        self.assertLess(progress["percent"], 100)

    def test_reference_progress_ignores_previous_stage_writes(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "references"
        app.APP.running_stage = "Reference Generation"
        app.APP.run_started_at = time.time() - 60
        app.APP.log = [
            "Wrote source frame 0000: source.png",
            "Wrote source frame 0001: source.png",
            "Wrote manifest: refs.csv",
            r"> python scripts\qwen_colorize_references.py --manifest refs.csv",
            "Rows: 11",
            "Colorize 0000: source.png -> color.png",
            "Queued ComfyUI prompt: prompt-id",
            "Wrote color.png",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "1/11 references")

    def test_colour_progress_ignores_previous_process_finished_lines(self) -> None:
        original_log = app.APP.log
        app.APP.running_stage_key = "colour"
        app.APP.running_stage = "Colorization"
        app.APP.run_started_at = time.time() - 60
        app.APP.log = [
            r"> python scripts\qwen_colorize_references.py --manifest refs.csv",
            "Wrote reference.png",
            "Process finished with exit code 0.",
            r"> python scripts\colorize_video.py --manifest refs.csv --method both",
            "Colorize segment 11/11 with deepexemplar: frames 1343-1440 using ref.png",
            "Wrote colorized video: deepexemplar.mp4",
            "Colorize segment 4/11 with colormnet: frames 527-632 using ref.png",
            "Sending prompt nodes: {'1': 'VHS_LoadVideo'}",
        ]

        try:
            progress = app.APP.estimate_running_progress()
        finally:
            app.APP.running_stage_key = ""
            app.APP.running_stage = ""
            app.APP.run_started_at = 0.0
            app.APP.log = original_log

        self.assertEqual(progress["label"], "Colorizing segment 4/11")

    def test_blank_project_defaults_outpainting_visible(self) -> None:
        with mock.patch.object(app, "SETTINGS_FILE", Path("missing-settings.json")), mock.patch.object(app, "newest", return_value=None):
            settings = app.load_settings()

        self.assertEqual(settings["global"]["expand_outpaint"], "true")
        self.assertEqual(settings["outpaint"]["target_height"], "source")
        self.assertEqual(settings["outpaint"]["seed_qwen_guides"], "false")

    def test_blank_loaded_project_defaults_outpainting_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "blank.arpp"
            payload = app.project_payload(app.APP.settings)
            payload["settings"] = {"global": {"source": "", "expand_outpaint": "false", "colorize": "true"}}
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = app.read_project_file(path)

        self.assertEqual(loaded["global"]["expand_outpaint"], "true")

    def test_section_preview_times_are_relative_to_trim_start(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        self.assertAlmostEqual(app.section_relative_seconds(app.APP.settings, 12), 0.0)
        self.assertAlmostEqual(app.section_relative_seconds(app.APP.settings, 18.5), 6.5)
        self.assertAlmostEqual(app.section_relative_seconds(app.APP.settings, 30), 12.0)

    def test_outpaint_chunks_prepares_section_before_reading_it(self) -> None:
        app.APP.settings["global"].update({"source": "input/example.mp4", "section_start": "12", "section_end": "24"})

        with mock.patch.object(server, "ensure_source_section_clip") as ensure, mock.patch.object(server, "resolve_video_source") as resolve_source:
            resolve_source.return_value = Path("missing-section.mp4")
            state = app.outpaint_chunks_state(app.APP.settings)

        ensure.assert_called_once_with(app.APP.settings)
        self.assertIn("not a readable file", state["error"])

    def test_media_clip_rejects_missing_source(self) -> None:
        with self.assertRaises(FileNotFoundError):
            app.media_clip_path(app.ROOT / "does-not-exist.mp4", 0, 1, "smoke")

    def test_preview_cache_name_uses_short_stem_and_hash(self) -> None:
        source = Path(r"C:\Users\mdamberger\AppData\Local\Programs\ai-remaster-pipeline\intermediate\source_sections\DrWho_Wheel_in_space_0000000000_0000154040.mp4")

        identity = app.aspect_preview_identity(source, 123, 456, "16:9", (8, 7, 0, 0), 0.0)
        preview = Path(r"C:\Users\mdamberger\AppData\Local\Programs\ai-remaster-pipeline\.cache\aspect_previews") / app.aid.artifact_name(
            app.aid.source_word(source.name),
            "aspectpreview",
            identity,
            "jpg",
        )

        self.assertTrue(preview.name.startswith("DrWho_aspectpreview_"))
        self.assertNotIn("16x9", preview.name)
        self.assertNotIn("crop", preview.name)
        self.assertLess(len(str(preview)), 240)

    def test_aspect_preview_cache_writes_signature_sidecar(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video placeholder")
            preview_dir = folder / "aspect_previews"
            identity = app.aspect_preview_identity(source, source.stat().st_size, source.stat().st_mtime_ns, "16:9", (8, 7, 0, 0), 0.0)
            target = preview_dir / app.aid.artifact_name(app.aid.source_word(source.name), "aspectpreview", identity, "jpg")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"preview")

            with mock.patch.dict(
                app.aspect_preview_cached.__globals__,
                {"ASPECT_PREVIEW_DIR": preview_dir, "extract_video_frame_at": mock.Mock(return_value=app.rel(target))},
            ):
                preview = app.aspect_preview_cached(str(source), source.stat().st_size, source.stat().st_mtime_ns, "16:9", (8, 7, 0, 0), 0.0)
                sidecar = json.loads((Path(app.resolve(preview)).with_suffix(".jpg.sig.json")).read_text(encoding="utf-8-sig"))

        self.assertTrue(Path(preview).name.startswith("source_aspectpreview_"))
        self.assertNotIn("16x9", Path(preview).name)
        self.assertEqual(sidecar["identity"]["aspect"], "16:9")
        self.assertEqual(sidecar["identity"]["crop"], [8, 7, 0, 0])

    def test_source_preview_analysis_regenerates_without_cache_clear_attribute(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video placeholder")
            preview_dir = folder / "previews"
            signature = (str(source), source.stat().st_size, source.stat().st_mtime_ns)

            def fake_generate_video_previews(_source, target_dir, progress=None, duration=None):
                target_dir.mkdir(parents=True, exist_ok=True)
                for index in range(app.source_previews_for_analysis.__globals__["SOURCE_PREVIEW_COUNT"]):
                    (target_dir / f"preview_{index}.jpg").write_bytes(b"preview")

            with mock.patch.dict(
                app.source_previews_for_analysis.__globals__,
                {"PREVIEW_DIR": preview_dir, "generate_video_previews": fake_generate_video_previews},
            ):
                previews = app.source_previews_for_analysis(signature, {"duration": "1"}, lambda _percent, _message: None)

        self.assertEqual(len(previews), app.source_previews_for_analysis.__globals__["SOURCE_PREVIEW_COUNT"])

    def test_frame_preview_reuses_fresh_existing_thumbnail(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video placeholder")
            target_dir = folder / "previews"
            target_dir.mkdir()
            target = target_dir / f"{app.safe_preview_name(source)}_thumb.jpg"
            target.write_bytes(b"cached thumbnail")

            with mock.patch.object(server, "local_tool", return_value="ffmpeg"), mock.patch.object(server.subprocess, "run") as run:
                self.assertEqual(app.extract_video_frame_at(source, target_dir, "thumb", 0), app.rel(target))

            run.assert_not_called()

    def test_files_for_skips_files_deleted_during_refresh(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            folder = Path(tmp_text)
            rel_folder = app.rel(folder)
            disappearing = folder / "vanishing.txt"
            disappearing.write_text("briefly here", encoding="utf-8")
            stage = app.Stage("smoke", "Smoke", "", (rel_folder,), (), ())
            real_stat = Path.stat
            calls = {"target": 0}

            def stat_once_then_missing(path: Path, *args, **kwargs):
                if path == disappearing:
                    calls["target"] += 1
                    if calls["target"] >= 2:
                        raise FileNotFoundError(str(path))
                return real_stat(path, *args, **kwargs)

            with mock.patch.object(Path, "stat", stat_once_then_missing):
                self.assertEqual(app.APP.files_for(stage), [])

    def test_state_endpoint_returns_json(self) -> None:
        server = app.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/state", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()

        self.assertIn("stages", payload)
        self.assertIn("settings", payload)

    def test_media_status_endpoint_reports_existing_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=app.ROOT) as tmp_text:
            media_file = Path(tmp_text) / "preview.png"
            media_file.write_bytes(b"preview")
            server = app.create_server("127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/api/media-status?path={urllib.parse.quote(app.rel(media_file))}"
                with urllib.request.urlopen(url, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["exists"])
        self.assertGreater(payload["mtime"], 0)

    def test_root_serves_static_frontend_shell(self) -> None:
        server = app.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as response:
                html = response.read().decode("utf-8")
        finally:
            server.shutdown()
            server.server_close()

        self.assertIn('/static/styles.css', html)
        self.assertIn('/static/js/core.js', html)
        self.assertIn('/static/js/render-cache.js', html)
        self.assertIn('/static/js/app.js', html)

    def test_quit_endpoint_acknowledges_and_stops_server(self) -> None:
        server = app.create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/api/quit",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            # request_quit() shuts the server down a moment after answering.
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        finally:
            server.shutdown()
            server.server_close()

    def test_quit_stops_running_stage_and_blocks_relaunch(self) -> None:
        class FakeRunningProcess:
            returncode = None

            def poll(self):
                return None

        previous_process = app.APP.process
        previous_quitting = app.APP.quitting
        app.APP.process = FakeRunningProcess()
        app.APP.quitting = False
        try:
            with mock.patch.object(server, "terminate_process_tree") as kill:
                app.APP.stop_for_quit()

            kill.assert_called_once_with(app.APP.process)
            self.assertTrue(app.APP.quitting)
            # While quitting, a Run All queue (or any caller) must not relaunch a stage.
            ok, message = app.APP.run_stage("outpaint")
            self.assertFalse(ok)
            self.assertIn("shutting down", message)
        finally:
            app.APP.process = previous_process
            app.APP.quitting = previous_quitting


if __name__ == "__main__":
    unittest.main()
