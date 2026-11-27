"""Opt-in GitHub auto-updater (`main` branch) & newest compatible system Python detector."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hfw_toolkit.config import ToolkitConfig

GITHUB_OWNER = "JP111555"
GITHUB_REPO = "HFWToolkit"
DEFAULT_UPDATE_BRANCH = "main"
MIN_PYTHON_VERSION = (3, 10)
VERSION_STAMP_FILE = ".hfw_version.json"

# Paths preserved verbatim during ZIP-based in-place updates
_PRESERVE_TOP_LEVEL: frozenset[str] = frozenset({
    ".venv",
    ".git",
    "cache",
    "exports",
    "bin",
    "dump",
    "hfw_toolkit_config.json",
    VERSION_STAMP_FILE,
})


def _parse_version_tuple(text: str) -> tuple[int, int, int] | None:
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def _is_windows_apps_alias(path: Path) -> bool:
    s = str(path).lower().replace("/", "\\")
    return "\\microsoft\\windowsapps\\" in s


_is_windows_store_stub = _is_windows_apps_alias


def _probe_python_exe(exe: Path, timeout: float = 3.0) -> dict[str, Any] | None:
    """Safely probe a candidate Python executable for version and architecture.

    Note: Fake Windows Store aliases in ``WindowsApps`` exit immediately with code 9009
    when invoked with ``-c`` (they never launch the Store UI when arguments are passed),
    while genuine Microsoft Store / MSIX Python Install Manager shims succeed cleanly.
    """
    try:
        if not exe.exists():
            return None
        res = subprocess.run(
            [
                str(exe),
                "-c",
                "import sys, platform; print(f'HFW_PY_OK|{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}|{platform.architecture()[0]}|{sys.executable}')",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if res.returncode != 0:
            return None
        lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip().startswith("HFW_PY_OK|")]
        if not lines:
            return None
        parts = lines[-1].split("|")
        if len(parts) < 3:
            return None
        vt = _parse_version_tuple(parts[1])
        if vt is None or vt[:2] < MIN_PYTHON_VERSION:
            return None
        resolved = Path(parts[3].strip()) if len(parts) >= 4 and parts[3].strip() else exe.resolve()
        return {
            "path": str(resolved),
            "version": f"{vt[0]}.{vt[1]}.{vt[2]}",
            "version_tuple": list(vt),
            "arch": parts[2].strip(),
            "compatible": True,
        }
    except Exception:
        return None


def detect_system_pythons(toolkit_dir: Path | None = None) -> dict[str, Any]:
    """Discover all compatible system Python 3.10+ installations (newest first) and compare against active Python."""
    candidates: list[Path] = []
    seen_raw: set[str] = set()

    def _add_candidate(p: Path | str | None) -> None:
        if not p:
            return
        try:
            pp = Path(str(p).strip().strip('"'))
            if not pp.exists():
                return
            # Skip the toolkit's own local .venv interpreter when scanning *system* Pythons
            if toolkit_dir is not None:
                try:
                    pp.resolve().relative_to((toolkit_dir / ".venv").resolve())
                    return
                except ValueError:
                    pass
            key = str(pp).lower() if _is_windows_apps_alias(pp) else str(pp.resolve()).lower()
            if key not in seen_raw:
                seen_raw.add(key)
                candidates.append(pp if _is_windows_apps_alias(pp) else pp.resolve())
        except Exception:
            pass

    # 1. Windows Python Launcher (`py -0p`)
    if shutil.which("py"):
        try:
            r = subprocess.run(
                ["py", "-0p"],
                capture_output=True,
                text=True,
                timeout=4.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if r.returncode == 0:
                for line in (r.stdout or "").splitlines():
                    # Format: " -V:3.14 *        C:\Users\...\python.exe" or " -3.13-64        C:\..."
                    m = re.search(r"([A-Za-z]:\\[^\r\n]+?python(?:\d+(?:\.\d+)?)?\.exe)", line, re.IGNORECASE)
                    if m:
                        _add_candidate(m.group(1))
        except Exception:
            pass

    # 2. Windows Registry (`PythonCore`)
    if sys.platform == "win32":
        try:
            import winreg

            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for sub_root in (r"SOFTWARE\Python\PythonCore", r"SOFTWARE\WOW6432Node\Python\PythonCore"):
                    try:
                        with winreg.OpenKey(hive, sub_root) as root_k:
                            idx = 0
                            while True:
                                try:
                                    ver_name = winreg.EnumKey(root_k, idx)
                                    idx += 1
                                except OSError:
                                    break
                                try:
                                    with winreg.OpenKey(root_k, f"{ver_name}\\InstallPath") as ip_k:
                                        exe_path, _ = winreg.QueryValueEx(ip_k, "ExecutablePath")
                                        _add_candidate(exe_path)
                                except OSError:
                                    try:
                                        with winreg.OpenKey(root_k, f"{ver_name}\\InstallPath") as ip_k:
                                            inst_dir, _ = winreg.QueryValueEx(ip_k, "")
                                            _add_candidate(Path(inst_dir) / "python.exe")
                                    except OSError:
                                        pass
                    except OSError:
                        pass
        except Exception:
            pass

        # 3. Standard Windows Python install directories (classic + PEP 773 Python Install Manager)
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            py_mgr_root = Path(local_app) / "Python"
            if py_mgr_root.is_dir():
                _add_candidate(py_mgr_root / "bin" / "python.exe")
                try:
                    for child in py_mgr_root.iterdir():
                        if child.is_dir() and re.match(r"^(?:pythoncore|python)[-_]?3", child.name, re.IGNORECASE):
                            _add_candidate(child / "python.exe")
                except OSError:
                    pass

        search_roots = [
            Path(local_app) / "Programs" / "Python" if local_app else None,
            Path(r"C:\Program Files"),
            Path(r"C:\Program Files (x86)"),
            Path(r"C:\\"),
        ]
        for root in search_roots:
            if root is None or not root.is_dir():
                continue
            try:
                for child in root.iterdir():
                    if child.is_dir() and re.match(r"^python3\d+$", child.name, re.IGNORECASE):
                        _add_candidate(child / "python.exe")
            except OSError:
                pass

    # 4. PATH (`where python` / `where python3` / `shutil.which`)
    if sys.platform == "win32":
        for cmd_name in ("python", "python3"):
            try:
                r = subprocess.run(
                    ["where", cmd_name],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if r.returncode == 0:
                    for line in (r.stdout or "").splitlines():
                        _add_candidate(line)
            except Exception:
                pass
    else:
        for nm in ("python3", "python"):
            w = shutil.which(nm)
            if w:
                _add_candidate(w)

    # Probe and sort newest first (prefer 64-bit on tie)
    probed: list[dict[str, Any]] = []
    seen_resolved: set[str] = set()
    for c in candidates:
        info = _probe_python_exe(c)
        if info is None:
            continue
        k = info["path"].lower()
        if k not in seen_resolved:
            seen_resolved.add(k)
            probed.append(info)

    probed.sort(
        key=lambda x: (tuple(x["version_tuple"]), 1 if "64" in x.get("arch", "") else 0),
        reverse=True,
    )

    cur_vt = (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    newest = probed[0] if probed else None
    upgrade_available = bool(newest and tuple(newest["version_tuple"]) > cur_vt)
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)

    return {
        "current_python": sys.executable,
        "current_version": f"{cur_vt[0]}.{cur_vt[1]}.{cur_vt[2]}",
        "current_version_tuple": list(cur_vt),
        "in_venv": in_venv,
        "newest_system_python": newest,
        "upgrade_available": upgrade_available,
        "installed_pythons": probed,
    }


def _run_git(toolkit_dir: Path, args: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=str(toolkit_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return res.returncode, (res.stdout or "").strip(), (res.stderr or "").strip()
    except Exception as exc:
        return -1, "", str(exc)


def get_local_version_info(toolkit_dir: Path) -> dict[str, Any]:
    """Inspect local install mode (`git` vs `zip`), current commit SHA, branch, and working-tree cleanliness."""
    git_dir = toolkit_dir / ".git"
    has_git = git_dir.exists() and bool(shutil.which("git"))
    if has_git:
        rc, sha, _ = _run_git(toolkit_dir, ["rev-parse", "HEAD"])
        _, branch, _ = _run_git(toolkit_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
        _, status_out, _ = _run_git(toolkit_dir, ["status", "--porcelain"])
        _, log_out, _ = _run_git(toolkit_dir, ["log", "-1", "--format=%cI|%s"])
        commit_date, commit_msg = ("", "")
        if "|" in log_out:
            commit_date, commit_msg = log_out.split("|", 1)
        if rc == 0 and sha:
            return {
                "install_mode": "git",
                "sha": sha,
                "short_sha": sha[:7],
                "branch": branch or DEFAULT_UPDATE_BRANCH,
                "dirty": bool(status_out.strip()),
                "commit_date": commit_date,
                "commit_message": commit_msg,
            }

    stamp = toolkit_dir / VERSION_STAMP_FILE
    if stamp.is_file():
        try:
            data = json.loads(stamp.read_text(encoding="utf-8"))
            sha = str(data.get("sha", ""))
            return {
                "install_mode": "zip",
                "sha": sha,
                "short_sha": sha[:7] if sha else "unknown",
                "branch": str(data.get("branch", DEFAULT_UPDATE_BRANCH)),
                "dirty": False,
                "commit_date": str(data.get("commit_date", "")),
                "commit_message": str(data.get("commit_message", "")),
            }
        except Exception:
            pass

    return {
        "install_mode": "zip",
        "sha": "",
        "short_sha": "local",
        "branch": DEFAULT_UPDATE_BRANCH,
        "dirty": False,
        "commit_date": "",
        "commit_message": "",
    }


def _fetch_remote_head(branch: str = DEFAULT_UPDATE_BRANCH, timeout: float = 4.5) -> dict[str, Any]:
    """Query GitHub REST API for the latest commit on `branch` (stdlib-only, offline-safe)."""
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits/{branch}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "HFWToolkit-Updater/1.2",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    sha = str(payload.get("sha", ""))
    commit = payload.get("commit") or {}
    committer = commit.get("committer") or commit.get("author") or {}
    msg = str(commit.get("message", "")).splitlines()[0] if commit.get("message") else ""
    return {
        "sha": sha,
        "short_sha": sha[:7] if sha else "",
        "branch": branch,
        "commit_date": str(committer.get("date", "")),
        "commit_message": msg,
        "html_url": str(payload.get("html_url", f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/commits/{branch}")),
    }


def check_for_updates(cfg: ToolkitConfig, branch: str | None = None, timeout: float = 4.5) -> dict[str, Any]:
    """Check GitHub `main` (or configured branch) against the local install without mutating anything."""
    target_branch = (branch or getattr(cfg, "update_branch", DEFAULT_UPDATE_BRANCH) or DEFAULT_UPDATE_BRANCH).strip()
    local = get_local_version_info(cfg.toolkit_dir)
    result: dict[str, Any] = {
        "repo": f"{GITHUB_OWNER}/{GITHUB_REPO}",
        "target_branch": target_branch,
        "auto_update": bool(getattr(cfg, "auto_update", False)),
        "check_updates_on_startup": bool(getattr(cfg, "check_updates_on_startup", True)),
        "local": local,
        "remote": None,
        "update_available": False,
        "ahead_of_remote": False,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }
    try:
        remote = _fetch_remote_head(target_branch, timeout=timeout)
        result["remote"] = remote
        r_sha = remote.get("sha", "")
        l_sha = local.get("sha", "")
        if r_sha and l_sha:
            if r_sha == l_sha:
                result["update_available"] = False
            elif local["install_mode"] == "git":
                # If remote commit is already an ancestor of local HEAD (e.g. Development branch ahead of main),
                # don't report a stale update!
                rc, _, _ = _run_git(cfg.toolkit_dir, ["merge-base", "--is-ancestor", r_sha, "HEAD"])
                if rc == 0:
                    result["update_available"] = False
                    result["ahead_of_remote"] = True
                else:
                    result["update_available"] = True
            else:
                result["update_available"] = True
        elif r_sha and not l_sha:
            # Unstamped ZIP download — allow one-click sync to latest main
            result["update_available"] = True
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        result["error"] = f"Could not reach GitHub ({exc})"
    return result


def apply_update(
    cfg: ToolkitConfig,
    branch: str | None = None,
    force: bool = False,
    reinstall_deps: bool = True,
    sync_addon: bool = True,
) -> dict[str, Any]:
    """Update HFWToolkit cleanly from GitHub (`main` branch by default) for both git clones and ZIP installs."""
    target_branch = (branch or getattr(cfg, "update_branch", DEFAULT_UPDATE_BRANCH) or DEFAULT_UPDATE_BRANCH).strip()
    tdir = cfg.toolkit_dir.resolve()
    local = get_local_version_info(tdir)
    steps_log: list[str] = []

    if local["install_mode"] == "git":
        if local["dirty"] and not force:
            return {
                "status": "blocked",
                "message": "Uncommitted local changes detected in git working tree. Commit/stash them first or pass --force.",
                "install_mode": "git",
                "steps": steps_log,
            }
        rc_f, out_f, err_f = _run_git(tdir, ["fetch", "origin", target_branch], timeout=45.0)
        if rc_f != 0:
            return {
                "status": "error",
                "message": f"git fetch origin {target_branch} failed: {err_f or out_f}",
                "install_mode": "git",
                "steps": steps_log,
            }
        steps_log.append(f"Fetched origin/{target_branch}")

        cur_branch = local.get("branch", "")
        if cur_branch == target_branch:
            rc_m, out_m, err_m = _run_git(tdir, ["merge", "--ff-only", f"origin/{target_branch}"], timeout=30.0)
            if rc_m != 0:
                return {
                    "status": "error",
                    "message": f"Could not fast-forward '{target_branch}' to origin/{target_branch}: {err_m or out_m}",
                    "install_mode": "git",
                    "steps": steps_log,
                }
            steps_log.append(out_m or f"Fast-forwarded {target_branch}")
        else:
            # On another branch (e.g. Development): only update if user explicitly passed --force to switch/merge
            if not force:
                return {
                    "status": "blocked",
                    "message": (
                        f"Local git checkout is on branch '{cur_branch}' (not '{target_branch}'). "
                        f"Switch to '{target_branch}' or pass --force to merge origin/{target_branch}."
                    ),
                    "install_mode": "git",
                    "steps": steps_log,
                }
            rc_m, out_m, err_m = _run_git(tdir, ["merge", "--ff-only", f"origin/{target_branch}"], timeout=30.0)
            if rc_m != 0:
                return {
                    "status": "error",
                    "message": f"git merge --ff-only origin/{target_branch} failed: {err_m or out_m}",
                    "install_mode": "git",
                    "steps": steps_log,
                }
            steps_log.append(out_m or f"Merged origin/{target_branch}")
        new_local = get_local_version_info(tdir)
    else:
        # ZIP install update: download branch archive and atomically replace code files while preserving user data!
        try:
            remote = _fetch_remote_head(target_branch, timeout=8.0)
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Could not query GitHub before download: {exc}",
                "install_mode": "zip",
                "steps": steps_log,
            }
        zip_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/archive/refs/heads/{target_branch}.zip"
        try:
            with tempfile.TemporaryDirectory(prefix="hfw_update_") as td:
                td_p = Path(td)
                zip_path = td_p / "update.zip"
                req = urllib.request.Request(zip_url, headers={"User-Agent": "HFWToolkit-Updater/1.2"})
                with urllib.request.urlopen(req, timeout=60.0) as resp, zip_path.open("wb") as wf:
                    shutil.copyfileobj(resp, wf)
                steps_log.append(f"Downloaded {target_branch}.zip from GitHub")

                extract_dir = td_p / "extracted"
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for member in zf.infolist():
                        # Zip-slip guard
                        target_member = (extract_dir / member.filename).resolve()
                        if not str(target_member).startswith(str(extract_dir.resolve())):
                            raise RuntimeError(f"Unsafe zip entry rejected: {member.filename}")
                    zf.extractall(extract_dir)

                roots = [p for p in extract_dir.iterdir() if p.is_dir()]
                if not roots:
                    raise RuntimeError("Downloaded archive had no root folder.")
                src_root = roots[0]

                updated_files = 0
                for child in src_root.iterdir():
                    if child.name in _PRESERVE_TOP_LEVEL:
                        continue
                    dst = tdir / child.name
                    if child.is_dir():
                        shutil.copytree(child, dst, dirs_exist_ok=True)
                        updated_files += 1
                    else:
                        shutil.copy2(child, dst)
                        updated_files += 1

                stamp_data = {
                    "sha": remote.get("sha", ""),
                    "branch": target_branch,
                    "commit_date": remote.get("commit_date", ""),
                    "commit_message": remote.get("commit_message", ""),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                (tdir / VERSION_STAMP_FILE).write_text(json.dumps(stamp_data, indent=2), encoding="utf-8")
                steps_log.append(f"Updated {updated_files} top-level item(s) while preserving config, .venv, cache & exports")
        except Exception as exc:
            return {
                "status": "error",
                "message": f"ZIP update failed: {exc}",
                "install_mode": "zip",
                "steps": steps_log,
            }
        new_local = get_local_version_info(tdir)

    # Refresh editable install / dependencies
    if reinstall_deps:
        try:
            r_pip = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(tdir), "--no-build-isolation", "--quiet"],
                capture_output=True,
                text=True,
                timeout=120.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if r_pip.returncode == 0:
                steps_log.append("Refreshed Python package dependencies")
            else:
                steps_log.append(f"pip refresh warning: {(r_pip.stderr or r_pip.stdout)[:160]}")
        except Exception as exc:
            steps_log.append(f"pip refresh skipped: {exc}")

    # Sync Blender add-on if installed
    if sync_addon:
        try:
            from hfw_toolkit.cli import install_blender_addon_files

            appdata = os.environ.get("APPDATA")
            if appdata and (Path(appdata) / "Blender Foundation" / "Blender").is_dir():
                installed = install_blender_addon_files(cfg)
                if installed:
                    steps_log.append(f"Synced Blender add-on ({len(installed)} version(s))")
        except Exception:
            pass

    return {
        "status": "updated",
        "message": f"Updated HFWToolkit to {new_local.get('short_sha', 'latest')} ({target_branch}). Restart HFW Studio to load new code.",
        "install_mode": local["install_mode"],
        "local": new_local,
        "restart_required": True,
        "steps": steps_log,
    }
