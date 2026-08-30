"""Safe setup, validation, status, and cleanup for coding-agent routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import statistics
import sys
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from codex_feedback_hook import release_session_lease
from feedback import (
    _state_generation,
    read_session_state,
    session_route_lock,
    session_state_path,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows lacks flock.
    fcntl = None  # type: ignore[assignment]


ROUTE_VERSION = 1
ROUTE_FILENAME = "route.json"
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


@dataclass(frozen=True)
class AgentAdapter:
    name: str
    settings_path: str
    hook_executable: str
    documentation: str
    contract: str


ADAPTERS = {
    "codex": AgentAdapter(
        name="codex",
        settings_path=".codex/hooks.json",
        hook_executable="quodet-codex-hook",
        documentation="https://learn.chatgpt.com/docs/hooks",
        contract="codex-hooks-2026-08-30",
    ),
    "claude": AgentAdapter(
        name="claude",
        settings_path=".claude/settings.json",
        hook_executable="quodet-claude-hook",
        documentation="https://code.claude.com/docs/en/hooks",
        contract="claude-code-post-tool-batch-2026-08-30",
    ),
}
LEGACY_AGENT_CONTRACTS = {
    "claude": {"claude-code-hooks-2026-08-30"},
}


@dataclass(frozen=True)
class RouteConfig:
    version: int
    agent: str
    root: str
    spool_dir: str
    session_id: str
    contract: str
    stop_grace_seconds: float

    @property
    def root_path(self) -> Path:
        return Path(self.root)

    @property
    def spool_path(self) -> Path:
        return Path(self.spool_dir)


def require_secure_platform() -> None:
    """Fail closed where Quodet cannot enforce its filesystem invariants."""
    required_flags = all(
        getattr(os, name, 0) for name in ("O_NOFOLLOW", "O_DIRECTORY")
    )
    descriptor_relative = all(
        operation in os.supports_dir_fd
        for operation in (os.link, os.mkdir, os.open, os.stat, os.unlink)
    )
    no_follow = os.link in os.supports_follow_symlinks
    if (
        os.name != "posix"
        or fcntl is None
        or not hasattr(os, "getuid")
        or not descriptor_relative
        or not no_follow
        or not required_flags
    ):
        raise RuntimeError(
            "agent feedback requires POSIX flock, descriptor-relative opens, "
            "O_NOFOLLOW, and owner-only Unix permissions"
        )


def _private_directory(path: Path) -> Path:
    existed = path.exists()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not existed:
        path.chmod(0o700)
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise PermissionError(f"directory must be owner-only: {path}")
    return path


def _private_file(path: Path) -> None:
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PermissionError(f"file must be owner-only: {path}")


def _create_private_json(path: Path, value: object) -> None:
    """Atomically create one private JSON file without replacing a winner."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".route-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class _SettingsLocation:
    root: Path
    root_descriptor: int
    parent_descriptor: int
    directory_chain: tuple[tuple[int, str, int], ...]
    filename: str

    def assert_attached(self) -> None:
        """Verify every held directory is still attached beneath the same root."""
        root_metadata = os.stat(self.root, follow_symlinks=False)
        held_root = os.fstat(self.root_descriptor)
        if (root_metadata.st_dev, root_metadata.st_ino) != (
            held_root.st_dev,
            held_root.st_ino,
        ):
            raise PermissionError("project root changed during settings setup")
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        for parent_descriptor, component, held_descriptor in self.directory_chain:
            try:
                current = os.open(component, flags, dir_fd=parent_descriptor)
            except OSError as error:
                raise PermissionError(
                    "settings directory changed during setup"
                ) from error
            try:
                current_metadata = os.fstat(current)
                held_metadata = os.fstat(held_descriptor)
                if (current_metadata.st_dev, current_metadata.st_ino) != (
                    held_metadata.st_dev,
                    held_metadata.st_ino,
                ):
                    raise PermissionError("settings directory changed during setup")
            finally:
                os.close(current)


@contextmanager
def _secure_settings_location(root: Path, relative_settings: str):
    """Open/create a real settings parent beneath root without following links."""
    relative = Path(relative_settings)
    if (
        relative.is_absolute()
        or len(relative.parts) < 2
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("agent settings path must be a nested relative path")
    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_descriptor = os.open(root, directory_flags)
    except OSError as error:
        raise PermissionError("project root must be a real directory") from error
    opened: list[int] = []
    chain: list[tuple[int, str, int]] = []
    current_descriptor = root_descriptor
    try:
        for component in relative.parts[:-1]:
            try:
                child_descriptor = os.open(
                    component, directory_flags, dir_fd=current_descriptor
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_descriptor)
                except FileExistsError:
                    pass
                try:
                    child_descriptor = os.open(
                        component, directory_flags, dir_fd=current_descriptor
                    )
                except OSError as error:
                    raise PermissionError(
                        "agent settings parent must be a real directory beneath root"
                    ) from error
            except OSError as error:
                raise PermissionError(
                    "agent settings parent must be a real directory beneath root"
                ) from error
            opened.append(child_descriptor)
            chain.append((current_descriptor, component, child_descriptor))
            current_descriptor = child_descriptor
        location = _SettingsLocation(
            root=root,
            root_descriptor=root_descriptor,
            parent_descriptor=current_descriptor,
            directory_chain=tuple(chain),
            filename=relative.parts[-1],
        )
        location.assert_attached()
        yield location
        location.assert_attached()
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)
        os.close(root_descriptor)


def _read_settings_json(location: _SettingsLocation) -> object | None:
    """Read one existing regular settings file without following a target link."""
    location.assert_attached()
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(
            location.filename, flags, dir_fd=location.parent_descriptor
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        raise PermissionError("agent settings target must be a regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PermissionError("agent settings target must be a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            value = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    location.assert_attached()
    return value


def _create_settings_json(location: _SettingsLocation, value: object) -> None:
    """Create settings relative to a held no-follow directory descriptor."""
    location.assert_attached()
    temporary_name = f".quodet-settings-{uuid.uuid4().hex}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(
        temporary_name,
        flags,
        0o600,
        dir_fd=location.parent_descriptor,
    )
    linked = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        location.assert_attached()
        os.link(
            temporary_name,
            location.filename,
            src_dir_fd=location.parent_descriptor,
            dst_dir_fd=location.parent_descriptor,
            follow_symlinks=False,
        )
        linked = True
        os.fsync(location.parent_descriptor)
        location.assert_attached()
    except BaseException:
        if linked:
            try:
                temporary_metadata = os.stat(
                    temporary_name,
                    dir_fd=location.parent_descriptor,
                    follow_symlinks=False,
                )
                target_metadata = os.stat(
                    location.filename,
                    dir_fd=location.parent_descriptor,
                    follow_symlinks=False,
                )
                if (temporary_metadata.st_dev, temporary_metadata.st_ino) == (
                    target_metadata.st_dev,
                    target_metadata.st_ino,
                ):
                    os.unlink(location.filename, dir_fd=location.parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=location.parent_descriptor)
        except FileNotFoundError:
            pass


def route_path(spool_dir: Path) -> Path:
    return spool_dir.expanduser().resolve() / ROUTE_FILENAME


def _validate_route_value(value: object) -> RouteConfig:
    legacy = {
        "version",
        "agent",
        "root",
        "spool_dir",
        "session_id",
        "contract",
    }
    if not isinstance(value, dict) or set(value) not in (
        legacy,
        legacy | {"stop_grace_seconds"},
    ):
        raise ValueError("route has unexpected or missing fields")
    if value["version"] != ROUTE_VERSION:
        raise ValueError("unsupported route version")
    agent = value["agent"]
    if not isinstance(agent, str) or agent not in ADAPTERS:
        raise ValueError("unsupported route agent")
    root = value["root"]
    spool = value["spool_dir"]
    session = value["session_id"]
    contract = value["contract"]
    stop_grace = value.get("stop_grace_seconds", 2.0)
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("route root must be absolute")
    if not isinstance(spool, str) or not Path(spool).is_absolute():
        raise ValueError("route spool_dir must be absolute")
    if not isinstance(session, str) or not SESSION_ID_RE.fullmatch(session):
        raise ValueError("route session_id is invalid")
    supported_contracts = {
        ADAPTERS[agent].contract,
        *LEGACY_AGENT_CONTRACTS.get(agent, set()),
    }
    if contract not in supported_contracts:
        raise ValueError("route contract does not match the selected agent")
    if (
        isinstance(stop_grace, bool)
        or not isinstance(stop_grace, (int, float))
        or not math.isfinite(float(stop_grace))
        or not 0 <= float(stop_grace) <= 10
    ):
        raise ValueError("route stop_grace_seconds must be between zero and ten")
    resolved_root = Path(root).resolve()
    resolved_spool = Path(spool).resolve()
    if os.fspath(resolved_root) != root or os.fspath(resolved_spool) != spool:
        raise ValueError("route paths must be canonical absolute paths")
    if (
        resolved_spool == resolved_root
        or resolved_root in resolved_spool.parents
        or resolved_spool in resolved_root.parents
    ):
        raise ValueError(
            "route spool must be outside and not an ancestor of the watched root"
        )
    return RouteConfig(
        version=ROUTE_VERSION,
        agent=agent,
        root=root,
        spool_dir=spool,
        session_id=session,
        contract=contract,
        stop_grace_seconds=float(stop_grace),
    )


def load_route(path: Path) -> RouteConfig:
    require_secure_platform()
    path = path.expanduser().resolve()
    _private_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid route file: {path}") from error
    route = _validate_route_value(value)
    if path != route_path(route.spool_path):
        raise ValueError("route file must be the configured spool's route.json")
    _private_directory(route.spool_path)
    return route


def validate_watcher_route(
    route: RouteConfig,
    *,
    root: Path,
    spool_dir: Path | None,
    session_id: str | None,
) -> tuple[Path, str]:
    actual_root = root.expanduser().resolve()
    if actual_root != route.root_path:
        raise ValueError("watch root does not match the agent route")
    if spool_dir is not None and spool_dir.expanduser().resolve() != route.spool_path:
        raise ValueError("--spool-dir does not match the agent route")
    if session_id is not None and session_id != route.session_id:
        raise ValueError("--session-id does not match the agent route")
    return route.spool_path, route.session_id


def _resolve_command(explicit: str | None, executable: str) -> str:
    candidate = explicit or shutil.which(executable)
    if not candidate:
        raise ValueError(f"{executable} is not installed or not on PATH")
    path = Path(candidate).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"hook command is not executable: {path}")
    return os.fspath(path)


def hook_configuration(
    route: RouteConfig,
    *,
    hook_command: str,
    agent_command: str,
    include_stop_grace: bool = True,
) -> dict[str, object]:
    common = (
        f"{shlex.quote(hook_command)} --spool-dir "
        f"{shlex.quote(route.spool_dir)} --session-id "
        f"{shlex.quote(route.session_id)} --root {shlex.quote(route.root)}"
    )
    cleanup = (
        f"{shlex.quote(agent_command)} cleanup --config "
        f"{shlex.quote(os.fspath(route_path(route.spool_path)))} --from-hook"
    )
    stop = f"{common} --event Stop"
    if include_stop_grace:
        stop += f" --stop-grace {route.stop_grace_seconds:g}"
    legacy_claude = route.contract in LEGACY_AGENT_CONTRACTS.get("claude", set())
    event = (
        "PostToolUse"
        if route.agent == "codex" or legacy_claude
        else "PostToolBatch"
    )
    edit_hook: dict[str, object] = {
        "hooks": [
            {
                "type": "command",
                "command": f"{common} --event {event}",
                "timeout": 10,
                "statusMessage": "Delivering Quodet feedback",
            }
        ],
    }
    if route.agent == "codex":
        edit_hook["matcher"] = "^(Write|Edit|apply_patch)$"
    elif legacy_claude:
        edit_hook["matcher"] = "^(Write|Edit)$"
    return {
        "hooks": {
            event: [edit_hook],
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": stop,
                            "timeout": max(
                                10, math.ceil(route.stop_grace_seconds) + 2
                            ),
                        }
                    ]
                }
            ],
            "SessionEnd": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": cleanup,
                            "timeout": 3,
                        }
                    ]
                }
            ],
        }
    }


def _settings_are_compatible(
    current: object,
    *,
    expected: object,
    legacy: object,
    stop_grace_seconds: float,
) -> bool:
    return current == expected or (
        stop_grace_seconds == 2.0 and current == legacy
    )


def _initialize_artifacts(
    route: RouteConfig,
    *,
    settings: Path,
    settings_location: _SettingsLocation,
    expected_settings: object,
    legacy_settings: object,
) -> tuple[Path, Path]:
    try:
        current_settings = _read_settings_json(settings_location)
    except PermissionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing settings are not valid JSON: {settings}") from error
    if current_settings is not None and not _settings_are_compatible(
        current_settings,
        expected=expected_settings,
        legacy=legacy_settings,
        stop_grace_seconds=route.stop_grace_seconds,
    ):
        raise FileExistsError(
            f"refusing to overwrite existing settings: {settings}; "
            "merge the generated hooks manually"
        )

    route_file = route_path(route.spool_path)
    if route_file.exists():
        existing_route = load_route(route_file)
        if existing_route != route:
            raise FileExistsError(f"refusing to overwrite existing route: {route_file}")

    _private_directory(route.spool_path)
    for name in (
        "pending", "claimed", "acknowledged", "rejected", "dedupe", "sessions",
        "metrics", "flush-hints", "flush-requests", "in-flight"
    ):
        _private_directory(route.spool_path / name)
    if not route_file.exists():
        try:
            _create_private_json(route_file, asdict(route))
        except FileExistsError:
            if load_route(route_file) != route:
                raise FileExistsError(
                    f"refusing to overwrite concurrently created route: {route_file}"
                ) from None
    if current_settings is None:
        try:
            _create_settings_json(settings_location, expected_settings)
        except FileExistsError:
            try:
                winner = _read_settings_json(settings_location)
            except PermissionError:
                raise
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FileExistsError(
                    f"refusing to overwrite concurrently created settings: {settings}"
                ) from error
            if winner != expected_settings:
                raise FileExistsError(
                    f"refusing to overwrite concurrently created settings: {settings}"
                ) from None
    final_settings = _read_settings_json(settings_location)
    if not _settings_are_compatible(
        final_settings,
        expected=expected_settings,
        legacy=legacy_settings,
        stop_grace_seconds=route.stop_grace_seconds,
    ):
        raise PermissionError("agent settings changed during setup")
    return route_file, settings


def initialize(
    agent: str,
    *,
    root: Path,
    spool_dir: Path,
    session_id: str,
    hook_command: str | None = None,
    agent_command: str | None = None,
    stop_grace_seconds: float = 2.0,
) -> tuple[Path, Path]:
    require_secure_platform()
    adapter = ADAPTERS[agent]
    canonical_root = root.expanduser().resolve()
    canonical_spool = spool_dir.expanduser().resolve()
    if not canonical_root.is_dir():
        raise ValueError(f"root is not a directory: {canonical_root}")
    route = _validate_route_value(
        {
            "version": ROUTE_VERSION,
            "agent": agent,
            "root": os.fspath(canonical_root),
            "spool_dir": os.fspath(canonical_spool),
            "session_id": session_id,
            "contract": adapter.contract,
            "stop_grace_seconds": stop_grace_seconds,
        }
    )
    resolved_hook = _resolve_command(hook_command, adapter.hook_executable)
    resolved_agent = _resolve_command(agent_command, "quodet-agent")
    settings = canonical_root / adapter.settings_path

    # Preserve validation-only reruns for an installed legacy contract while
    # generating the current contract for every new route.
    existing_route_file = route_path(canonical_spool)
    if existing_route_file.exists():
        existing_route = load_route(existing_route_file)
        requested_with_existing_contract = replace(
            route, contract=existing_route.contract
        )
        if existing_route != requested_with_existing_contract:
            raise FileExistsError(
                f"refusing to overwrite existing route: {existing_route_file}"
            )
        route = existing_route
    expected_settings = hook_configuration(
        route, hook_command=resolved_hook, agent_command=resolved_agent
    )
    legacy_settings = hook_configuration(
        route,
        hook_command=resolved_hook,
        agent_command=resolved_agent,
        include_stop_grace=False,
    )

    # Preserve validation-only behavior when another route already owns the
    # requested spool: do not even create an empty agent settings directory.
    # Open every settings-path component relative to the canonical root without
    # following links and retain those descriptors through validation/creation.
    with _secure_settings_location(canonical_root, adapter.settings_path) as location:
        return _initialize_artifacts(
            route,
            settings=settings,
            settings_location=location,
            expected_settings=expected_settings,
            legacy_settings=legacy_settings,
        )


def _session_lease_path(route: RouteConfig) -> Path:
    return session_state_path(
        route.spool_path,
        root=route.root_path,
        configured_session_id=route.session_id,
    )


def _producer_lease_path(route: RouteConfig) -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path(tempfile.gettempdir())
    root_key = hashlib.sha256(route.root.encode()).hexdigest()
    return base / f"quodet-{os.getuid()}" / "active-roots" / f"{root_key}.lock"


def _producer_active(route: RouteConfig) -> bool:
    lease = _producer_lease_path(route)
    if not lease.exists():
        return False
    stream = lease.open("a+", encoding="utf-8")
    try:
        try:
            assert fcntl is not None
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        return False
    finally:
        stream.close()


@contextmanager
def _exclusive_cleanup_lease(route: RouteConfig):
    """Exclude watcher startup/publication for the entire cleanup operation."""
    require_secure_platform()
    lease = _producer_lease_path(route)
    _private_directory(lease.parent)
    stream = lease.open("a+", encoding="utf-8")
    lease.chmod(0o600)
    try:
        metadata = lease.stat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise PermissionError("producer lease must be owner-only")
        assert fcntl is not None
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("cannot clean a route while its watcher is active") from error
        yield
    finally:
        stream.close()


def _payload_owned(path: Path, route: RouteConfig) -> bool:
    try:
        _private_file(path)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, PermissionError):
        return False
    return (
        isinstance(value, dict)
        and value.get("root") == route.root
        and value.get("session_id") == route.session_id
    )


def route_status(route: RouteConfig) -> dict[str, object]:
    _private_directory(route.spool_path)
    counts: dict[str, int] = {}
    for name in (
        "pending",
        "claimed",
        "acknowledged",
        "rejected",
        "dedupe",
        "flush-hints",
        "flush-requests",
        "in-flight",
    ):
        directory = _private_directory(route.spool_path / name)
        counts[name] = sum(_payload_owned(path, route) for path in directory.glob("*.json"))
    lease = _session_lease_path(route)
    bound_agent_session: str | None = None
    session_state = "unbound"
    session_generation = 0
    with session_route_lock(
        route.spool_path,
        root=route.root_path,
        configured_session_id=route.session_id,
        exclusive=False,
    ):
        if lease.exists():
            if not _payload_owned_lease(lease, route):
                raise PermissionError("session lease ownership does not match route")
            value = read_session_state(
                route.spool_path,
                root=route.root_path,
                configured_session_id=route.session_id,
            )
            session_state, session_generation = _state_generation(
                value,
                root=route.root_path,
                configured_session_id=route.session_id,
            )
            if session_state == "bound" and value is not None:
                bound_agent_session = value["codex_session_id"]  # type: ignore[assignment]
    latency = _latency_summary(route)
    return {
        "agent": route.agent,
        "root": route.root,
        "spool_dir": route.spool_dir,
        "session_id": route.session_id,
        "contract": route.contract,
        "platform_secure": True,
        "producer_active": _producer_active(route),
        "session_state": session_state,
        "session_generation": session_generation,
        "bound_agent_session": bound_agent_session,
        "feedback": counts,
        "latency_ms": latency,
    }


def _latency_summary(route: RouteConfig) -> dict[str, object]:
    fields = (
        "debounce_ms",
        "detection_to_flush_ms",
        "flush_to_provider_ms",
        "provider_ms",
        "publication_ms",
        "hook_wait_ms",
        "hook_execution_ms",
        "total_edit_to_feedback_ms",
    )
    samples: dict[str, list[float]] = {field: [] for field in fields}
    metrics = _private_directory(route.spool_path / "metrics")
    for path in metrics.glob("*.json"):
        if not _payload_owned(path, route):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for field in fields:
            sample = value.get(field)
            if isinstance(sample, (int, float)) and not isinstance(sample, bool):
                samples[field].append(float(sample))
    result: dict[str, object] = {"samples": max((len(v) for v in samples.values()), default=0)}
    for field, values in samples.items():
        if not values:
            result[field] = {"median": None, "p95": None}
            continue
        ordered = sorted(values)
        p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
        result[field] = {
            "median": statistics.median(ordered),
            "p95": ordered[p95_index],
        }
    return result


def _payload_owned_lease(path: Path, route: RouteConfig) -> bool:
    try:
        _private_file(path)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, PermissionError):
        return False
    if not isinstance(value, dict):
        return False
    try:
        state, _ = _state_generation(
            value, root=route.root_path, configured_session_id=route.session_id
        )
    except ValueError:
        return False
    return state == "closed" or isinstance(value.get("codex_session_id"), str)


def cleanup_route(
    route: RouteConfig,
    *,
    agent_session_id: str | None,
    discard_feedback: bool = False,
) -> dict[str, int]:
    """Clean one exact inactive route without touching any other session."""
    with _exclusive_cleanup_lease(route):
        with session_route_lock(
            route.spool_path,
            root=route.root_path,
            configured_session_id=route.session_id,
            exclusive=True,
        ):
            lease = _session_lease_path(route)
            if lease.exists():
                if not _payload_owned_lease(lease, route):
                    raise PermissionError("session lease ownership does not match route")
                lease_value = read_session_state(
                    route.spool_path,
                    root=route.root_path,
                    configured_session_id=route.session_id,
                )
                lease_state, _ = _state_generation(
                    lease_value,
                    root=route.root_path,
                    configured_session_id=route.session_id,
                )
                if (
                    lease_state == "bound"
                    and lease_value is not None
                    and agent_session_id != lease_value["codex_session_id"]
                ):
                    raise PermissionError(
                        "agent session identity does not match route lease"
                    )
            active = sum(
                _payload_owned(path, route)
                for state in ("pending", "claimed")
                for path in (route.spool_path / state).glob("*.json")
            )
            if active and not discard_feedback:
                raise RuntimeError(
                    "route still has pending or claimed feedback; rerun with "
                    "--discard-feedback only after the agent session ends"
                )
            removed: dict[str, int] = {}
            # Latency/protocol metrics intentionally survive route cleanup so operators
            # can inspect completed-session medians and tails. They contain no source or
            # finding text and remain protected by the private route spool.
            states = (
                "pending",
                "claimed",
                "acknowledged",
                "rejected",
                "dedupe",
                "flush-hints",
                "flush-requests",
                "in-flight",
            )
            for name in states:
                removed[name] = 0
                for path in (route.spool_path / name).glob("*.json"):
                    if _payload_owned(path, route):
                        path.unlink()
                        removed[name] += 1
            policy = route.spool_path / "policy" / "rounds.json"
            if policy.exists():
                _private_file(policy)
                policy.unlink()
                removed["policy"] = 1
            else:
                removed["policy"] = 0
            if lease.exists():
                lease.unlink()
                removed["session_lease"] = 1
            else:
                removed["session_lease"] = 0
            return removed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create or validate one agent route")
    init.add_argument("agent", choices=tuple(ADAPTERS))
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--spool-dir", type=Path, required=True)
    init.add_argument("--session-id", required=True)
    init.add_argument("--hook-command")
    init.add_argument("--agent-command")
    init.add_argument("--stop-grace", type=float, default=2.0)
    for name in ("status", "cleanup"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--json", action="store_true")
    cleanup = commands.choices["cleanup"]
    cleanup.add_argument("--discard-feedback", action="store_true")
    cleanup.add_argument("--from-hook", action="store_true", help=argparse.SUPPRESS)
    cleanup.add_argument("--agent-session-id")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "init":
            route, settings = initialize(
                args.agent,
                root=args.root,
                spool_dir=args.spool_dir,
                session_id=args.session_id,
                hook_command=args.hook_command,
                agent_command=args.agent_command,
                stop_grace_seconds=args.stop_grace,
            )
            print(json.dumps({"route": os.fspath(route), "settings": os.fspath(settings)}))
            return 0
        route = load_route(args.config)
        if args.command == "status":
            value = route_status(route)
        else:
            agent_session_id = args.agent_session_id
            if args.from_hook:
                try:
                    hook_input = json.load(sys.stdin)
                except json.JSONDecodeError:
                    hook_input = {}
                if not isinstance(hook_input, dict):
                    hook_input = {}
                hook_cwd = hook_input.get("cwd")
                valid_hook_route = (
                    hook_input.get("hook_event_name") == "SessionEnd"
                    and isinstance(hook_cwd, str)
                    and Path(hook_cwd).resolve() == route.root_path
                )
                agent_session_id = (
                    hook_input.get("session_id") if valid_hook_route else None
                )
                value = {
                    "session_released": release_session_lease(
                        route.spool_path,
                        root=route.root_path,
                        configured_session_id=route.session_id,
                        agent_session_id=agent_session_id,
                    )
                }
            else:
                value = cleanup_route(
                    route,
                    agent_session_id=agent_session_id,
                    discard_feedback=args.discard_feedback,
                )
        print(json.dumps(value, indent=None if args.json else 2, sort_keys=True))
        return 0
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
