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
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

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
        contract="claude-code-hooks-2026-08-30",
    ),
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
    if (
        os.name != "posix"
        or fcntl is None
        or not hasattr(os, "getuid")
        or os.open not in os.supports_dir_fd
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
    if contract != ADAPTERS[agent].contract:
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
    matcher = "^(Write|Edit|apply_patch)$" if route.agent == "codex" else "^(Write|Edit)$"
    return {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": matcher,
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{common} --event PostToolUse",
                            "timeout": 10,
                            "statusMessage": "Delivering Quodet feedback",
                        }
                    ],
                }
            ],
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
    expected_settings = hook_configuration(
        route, hook_command=resolved_hook, agent_command=resolved_agent
    )
    legacy_settings = hook_configuration(
        route,
        hook_command=resolved_hook,
        agent_command=resolved_agent,
        include_stop_grace=False,
    )

    # Validate every collision before creating either artifact. Existing agent
    # settings are never merged or overwritten implicitly.
    if settings.exists():
        try:
            current_settings = json.loads(settings.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"existing settings are not valid JSON: {settings}") from error
        if current_settings not in (expected_settings, legacy_settings):
            raise FileExistsError(
                f"refusing to overwrite existing settings: {settings}; "
                "merge the generated hooks manually"
            )
    route_file = route_path(canonical_spool)
    if route_file.exists():
        existing_route = load_route(route_file)
        if existing_route != route:
            raise FileExistsError(f"refusing to overwrite existing route: {route_file}")

    _private_directory(canonical_spool)
    for name in (
        "pending", "claimed", "acknowledged", "rejected", "dedupe", "sessions",
        "metrics", "flush-hints", "in-flight"
    ):
        _private_directory(canonical_spool / name)
    if not route_file.exists():
        try:
            _create_private_json(route_file, asdict(route))
        except FileExistsError:
            if load_route(route_file) != route:
                raise FileExistsError(
                    f"refusing to overwrite concurrently created route: {route_file}"
                ) from None
    if not settings.exists():
        settings.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            _create_private_json(settings, expected_settings)
        except FileExistsError:
            try:
                winner = json.loads(settings.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FileExistsError(
                    f"refusing to overwrite concurrently created settings: {settings}"
                ) from error
            if winner != expected_settings:
                raise FileExistsError(
                    f"refusing to overwrite concurrently created settings: {settings}"
                ) from None
    return route_file, settings


def _session_lease_path(route: RouteConfig) -> Path:
    identity = f"{route.root}\0{route.session_id}".encode()
    return route.spool_path / "sessions" / f"{hashlib.sha256(identity).hexdigest()}.json"


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
        "in-flight",
    ):
        directory = _private_directory(route.spool_path / name)
        counts[name] = sum(_payload_owned(path, route) for path in directory.glob("*.json"))
    lease = _session_lease_path(route)
    bound_agent_session: str | None = None
    if lease.exists() and _payload_owned_lease(lease, route):
        value = json.loads(lease.read_text(encoding="utf-8"))
        bound_agent_session = value["codex_session_id"]
    latency = _latency_summary(route)
    return {
        "agent": route.agent,
        "root": route.root,
        "spool_dir": route.spool_dir,
        "session_id": route.session_id,
        "contract": route.contract,
        "platform_secure": True,
        "producer_active": _producer_active(route),
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
    return (
        isinstance(value, dict)
        and value.get("root") == route.root
        and value.get("configured_session_id") == route.session_id
        and isinstance(value.get("codex_session_id"), str)
    )


def cleanup_route(
    route: RouteConfig,
    *,
    agent_session_id: str | None,
    discard_feedback: bool = False,
) -> dict[str, int]:
    """Clean one exact inactive route without touching any other session."""
    with _exclusive_cleanup_lease(route):
        lease = _session_lease_path(route)
        if lease.exists():
            if not _payload_owned_lease(lease, route):
                raise PermissionError("session lease ownership does not match route")
            lease_value = json.loads(lease.read_text(encoding="utf-8"))
            if agent_session_id != lease_value["codex_session_id"]:
                raise PermissionError("agent session identity does not match route lease")
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
            "in-flight",
        )
        for name in states:
            removed[name] = 0
            for path in (route.spool_path / name).glob("*.json"):
                if _payload_owned(path, route):
                    path.unlink()
                    removed[name] += 1
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
                agent_session_id = hook_input.get("session_id")
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
