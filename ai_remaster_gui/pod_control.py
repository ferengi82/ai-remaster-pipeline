"""Best-effort RunPod pod shutdown after a stage finishes.

Enabled by ARP_SHUTDOWN_AFTER_UPSCALE=stop|terminate (empty = disabled). Used so an unattended
multi-GPU upscale job can power the pod down when it's done instead of billing idle GPUs.

Mechanism: if RUNPOD_API_KEY is set, call the RunPod GraphQL API (stdlib urllib); otherwise fall
back to the on-pod `runpodctl` CLI. Everything is wrapped so a failure only logs — it never raises
into the caller (the upscale result must not depend on the shutdown succeeding).
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from typing import Callable

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"


def shutdown_pod(action: str, log: Callable[[str], None] = print) -> bool:
    """Stop or terminate the current RunPod pod. Returns True if a shutdown was triggered.

    action: "stop" (pause, keep pod+volume) or "terminate" (delete pod; volume data survives).
    Any other value (incl. "") is a no-op.
    """
    action = (action or "").strip().lower()
    if action not in ("stop", "terminate"):
        return False

    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    if not pod_id:
        log("[shutdown] RUNPOD_POD_ID is not set — cannot shut the pod down (skipping).")
        return False

    log(f"[shutdown] Upscale finished — requesting pod {action} for {pod_id} …")
    api_key = (os.environ.get("RUNPOD_API_KEY") or os.environ.get("ARP_RUNPOD_API_KEY") or "").strip()
    if api_key and _shutdown_via_api(action, pod_id, api_key, log):
        return True
    return _shutdown_via_cli(action, pod_id, log)


def _shutdown_via_api(action: str, pod_id: str, api_key: str, log: Callable[[str], None]) -> bool:
    if action == "stop":
        query = 'mutation { podStop(input: {podId: "%s"}) { id desiredStatus } }' % pod_id
    else:
        query = 'mutation { podTerminate(input: {podId: "%s"}) }' % pod_id
    data = json.dumps({"query": query}).encode("utf-8")
    request = urllib.request.Request(
        f"{RUNPOD_GRAPHQL}?api_key={api_key}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - best effort, fall back to CLI
        log(f"[shutdown] RunPod API call failed ({exc}); trying runpodctl …")
        return False
    if body.get("errors"):
        log(f"[shutdown] RunPod API error: {body['errors']}; trying runpodctl …")
        return False
    log(f"[shutdown] Pod {action} triggered via RunPod API.")
    return True


def _shutdown_via_cli(action: str, pod_id: str, log: Callable[[str], None]) -> bool:
    verb = "stop" if action == "stop" else "remove"
    try:
        subprocess.run(["runpodctl", verb, "pod", pod_id], check=True, timeout=60)
    except Exception as exc:  # noqa: BLE001 - best effort
        log(f"[shutdown] runpodctl {verb} failed ({exc}). Pod left running; "
            f"set RUNPOD_API_KEY for a reliable shutdown.")
        return False
    log(f"[shutdown] Pod {action} triggered via runpodctl.")
    return True
