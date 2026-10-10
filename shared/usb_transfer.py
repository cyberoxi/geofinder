"""Copy GeoFinder deployment packages to/from USB flash drives.

Desktop writes ``geofinder_packages/pkg_*.zip`` onto a removable drive.
Jetson watches the same folder and imports into a local packages directory.
"""

from __future__ import annotations

import os
import platform
import shutil
import string
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

USB_PACKAGE_DIR = "geofinder_packages"
PACKAGE_GLOB = "pkg_*.zip"


def list_removable_drives() -> List[Path]:
    """Return mount roots that look like removable USB media."""
    system = platform.system()
    if system == "Windows":
        return _windows_removable()
    return _linux_removable()


def latest_package_zip(packages_dir: str | Path) -> Optional[Path]:
    """Newest ``pkg_*.zip`` under a project packages folder."""
    root = Path(packages_dir)
    if not root.is_dir():
        return None
    zips = sorted(root.glob(PACKAGE_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def copy_package_to_usb(
    zip_path: str | Path,
    drive: str | Path,
    *,
    subdir: str = USB_PACKAGE_DIR,
) -> Path:
    """Copy a package zip into ``<drive>/<subdir>/`` and return the destination path."""
    src = Path(zip_path)
    if not src.is_file() or src.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Package zip not found: {src}")
    root = Path(drive)
    if not root.exists():
        raise FileNotFoundError(f"USB drive not found: {root}")
    dest_dir = root / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return dest


def find_usb_packages(
    drives: Optional[Sequence[str | Path]] = None,
    *,
    subdir: str = USB_PACKAGE_DIR,
) -> List[Path]:
    """All ``pkg_*.zip`` files found on removable drives (newest first)."""
    roots = [Path(d) for d in drives] if drives is not None else list_removable_drives()
    found: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for folder in (root / subdir, root):
            if not folder.is_dir():
                continue
            found.extend(folder.glob(PACKAGE_GLOB))
    # Deduplicate by resolve, keep newest
    by_key = {}
    for p in found:
        try:
            key = p.resolve()
        except OSError:
            key = p
        by_key[key] = p
    return sorted(by_key.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def import_package_from_usb(
    zip_path: str | Path,
    dest_dir: str | Path,
) -> Path:
    """Copy a USB package into a local directory; return the local path."""
    src = Path(zip_path)
    if not src.is_file():
        raise FileNotFoundError(f"USB package not found: {src}")
    out = Path(dest_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / src.name
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    return dest


def pick_drive_for_copy(drives: Optional[Sequence[Path]] = None) -> Optional[Path]:
    """Single removable drive if exactly one is present; else None (caller should ask)."""
    items = list(drives) if drives is not None else list_removable_drives()
    if len(items) == 1:
        return items[0]
    return None


def format_drive_label(drive: Path) -> str:
    return str(drive)


def drive_choices(drives: Optional[Iterable[Path]] = None) -> List[Tuple[str, Path]]:
    items = list(drives) if drives is not None else list_removable_drives()
    return [(format_drive_label(d), d) for d in items]


def _windows_removable() -> List[Path]:
    drives: List[Path] = []
    try:
        import ctypes

        GetDriveTypeW = ctypes.windll.kernel32.GetDriveTypeW  # type: ignore[attr-defined]
        DRIVE_REMOVABLE = 2
        for letter in string.ascii_uppercase:
            root = f"{letter}:\\"
            if GetDriveTypeW(root) == DRIVE_REMOVABLE and os.path.exists(root):
                drives.append(Path(root))
    except Exception:
        # Fallback: any existing drive letter that is not C:
        for letter in string.ascii_uppercase:
            if letter == "C":
                continue
            root = Path(f"{letter}:/")
            if root.exists():
                drives.append(root)
    return drives


def _linux_removable() -> List[Path]:
    drives: List[Path] = []
    seen = set()

    def _add(path: Path) -> None:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen or not path.is_dir():
            return
        seen.add(key)
        drives.append(path)

    for base in (Path("/media"), Path("/run/media")):
        if not base.is_dir():
            continue
        for user_dir in sorted(base.iterdir()):
            if not user_dir.is_dir():
                continue
            for mount in sorted(user_dir.iterdir()):
                if mount.is_dir():
                    _add(mount)

    mnt = Path("/mnt")
    if mnt.is_dir():
        for mount in sorted(mnt.iterdir()):
            if mount.is_dir() and any(mount.iterdir()):
                _add(mount)

    # Also parse /proc/mounts for vfat/exfat/ntfs under media paths
    mounts_file = Path("/proc/mounts")
    if mounts_file.exists():
        try:
            for line in mounts_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                mp, fstype = parts[1], parts[2]
                if fstype not in {"vfat", "exfat", "ntfs", "fuseblk", "msdos"}:
                    continue
                if mp.startswith(("/media/", "/run/media/", "/mnt/")):
                    _add(Path(mp))
        except OSError:
            pass

    return drives
