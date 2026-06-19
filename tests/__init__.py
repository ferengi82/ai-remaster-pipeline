"""Smoke tests for ARP."""

import os
import tempfile
from pathlib import Path

# Point the whole suite at a throwaway settings file so tests neither read the developer's live
# .ai_remaster_gui.json (which would make assertions depend on local UI state) nor clobber it.
# `unittest discover` imports this package before any test module imports ai_remaster_gui.config
# (where the path is resolved), so this redirect always wins the import race regardless of which
# test module loads the package first.
os.environ.setdefault("ARP_SETTINGS_FILE", str(Path(tempfile.mkdtemp(prefix="arp-test-settings-")) / "settings.json"))
