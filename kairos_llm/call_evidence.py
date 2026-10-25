"""Opt-in create-only application-boundary evidence for budgeted LLM calls.

Records contain the exact UTF-8 system/user/schema values handed to the
gateway and the exact UTF-8 ``LLMResult.content`` returned by it. They are not
provider HTTP request/response bodies, provider-authenticated evidence, or
authorization to use a result. Credentials, headers, and environment values are
never accepted by this module.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, cast

from .schemas import LLMResult

MAX_CALL_RECORD_BYTES = 4_000_000
MAX_APPLICATION_INPUT_BYTES = 2_900_000
_CALL_ID = re.compile(r"^[a-f0-9]{32}$")
_PHASES = frozenset({"started", "completed"})


class LLMCallEvidenceSink(Protocol):
    """Explicit sink; implementations must be create-only and conflict-safe."""

    def create_only(self, call_id: str, phase: str, record: bytes) -> None: ...


class EvidenceConflictError(ValueError):
    """A call ID already names different immutable evidence."""


def emit_call_evidence(sink: LLMCallEvidenceSink, call_id: str, phase: str, record: bytes) -> None:
    """A synchronous acknowledgement is mandatory before a call/result proceeds."""
    writer = cast(Callable[..., object], sink.create_only)
    outcome = writer(call_id, phase, record)
    if inspect.iscoroutine(outcome):
        outcome.close()
    if outcome is not None:
        raise ValueError("evidence sink must synchronously acknowledge persistence without a return value")


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _encoded(data: bytes) -> dict[str, str]:
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "base64": base64.b64encode(data).decode("ascii"),
    }


def preflight_call_evidence(*, system: str, user: str, schema_bytes: bytes) -> None:
    """Refuse oversized input before reserving budget or invoking a provider."""
    if not isinstance(system, str) or not isinstance(user, str) or not isinstance(schema_bytes, bytes):
        raise TypeError("exact application prompt strings and schema bytes are required")
    raw_size = len(system.encode("utf-8")) + len(user.encode("utf-8")) + len(schema_bytes)
    if raw_size > MAX_APPLICATION_INPUT_BYTES:
        raise ValueError("application call evidence input exceeds the bounded preflight size")


def _base_record(*, call_id: str, workload: str, effort: str, system: str, user: str, schema_bytes: bytes):
    system_bytes = system.encode("utf-8")
    user_bytes = user.encode("utf-8")
    return {
        "version": "kairos.llm-application-call-evidence.v1",
        "capture_scope": "application-boundary-not-provider-http",
        "call_id": call_id,
        "workload": workload,
        "effort": effort,
        "request": {
            "system_utf8": _encoded(system_bytes),
            "user_utf8": _encoded(user_bytes),
            "schema_canonical_json_utf8": _encoded(schema_bytes),
        },
    }


def build_call_attempt_record(
    *,
    call_id: str,
    system: str,
    user: str,
    schema_bytes: bytes,
    workload: str,
    effort: str,
    provider: str,
    model: str,
    reservation_id: str,
    reserved_microusd: int,
    attempt_started_at_ms: int,
) -> bytes:
    """Serialize the original prompt and hold immediately before one provider call."""
    if not _CALL_ID.fullmatch(call_id):
        raise ValueError("call_id must be a lowercase UUID hex string")
    preflight_call_evidence(system=system, user=user, schema_bytes=schema_bytes)
    for name, value in (("provider", provider), ("model", model), ("reservation_id", reservation_id)):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"completed call omitted mandatory {name}")
    if type(reserved_microusd) is not int or reserved_microusd < 0:
        raise ValueError("exact reserved budget amount is required")
    if type(attempt_started_at_ms) is not int or attempt_started_at_ms < 0:
        raise ValueError("original gateway attempt clock is required")
    record = _canonical_json(
        {
            **_base_record(
                call_id=call_id,
                workload=workload,
                effort=effort,
                system=system,
                user=user,
                schema_bytes=schema_bytes,
            ),
            "phase": "ATTEMPT_STARTED",
            "provider_call": {
                "provider": provider,
                "model": model,
                "attempt_started_at_ms": attempt_started_at_ms,
            },
            "budget": {
                "reservation_id": reservation_id,
                "status": "RESERVED",
                "reserved_microusd": reserved_microusd,
            },
        }
    )
    if len(record) > MAX_CALL_RECORD_BYTES:
        raise ValueError("application call evidence exceeds the bounded record size")
    return record


def build_call_completion_record(
    *,
    call_id: str,
    attempt_record: bytes,
    result: LLMResult,
    reserved_microusd: int,
    actual_microusd: int,
    budget_commit_observed_at_ms: int,
) -> bytes:
    """Serialize an exact application reply after the durable budget commit."""
    if not _CALL_ID.fullmatch(call_id):
        raise ValueError("call_id must be a lowercase UUID hex string")
    if not isinstance(attempt_record, bytes):
        raise TypeError("the exact started record bytes are required")
    if len(attempt_record) > MAX_CALL_RECORD_BYTES:
        raise ValueError("started call evidence exceeds the bounded record size")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("started evidence has duplicate fields")
            value[key] = item
        return value

    try:
        attempt = json.loads(attempt_record, object_pairs_hook=unique_object)
        if (
            attempt["version"] != "kairos.llm-application-call-evidence.v1"
            or attempt["phase"] != "ATTEMPT_STARTED"
            or attempt["call_id"] != call_id
            or attempt["provider_call"]["provider"] != result.provider
            or attempt["provider_call"]["model"] != result.model
            or attempt["provider_call"]["attempt_started_at_ms"] != result.attempt_started_at_ts_ms
            or attempt["budget"]["status"] != "RESERVED"
            or attempt["budget"]["reservation_id"] != result.budget_reservation_id
            or attempt["budget"]["reserved_microusd"] != reserved_microusd
        ):
            raise ValueError
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise ValueError("completion does not bind the exact original attempt and budget hold") from None
    if type(result.cached) is not bool:
        raise ValueError("gateway input-cache metadata must be boolean")
    for name in ("provider", "model", "request_id", "budget_reservation_id"):
        value = getattr(result, name, None)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"completed call omitted mandatory {name}")
    if not isinstance(result.content, str):
        raise ValueError("completed call content must be text")
    if (
        type(reserved_microusd) is not int
        or type(actual_microusd) is not int
        or not 0 <= actual_microusd <= reserved_microusd
        or type(budget_commit_observed_at_ms) is not int
        or budget_commit_observed_at_ms < 0
    ):
        raise ValueError("exact committed and reserved budget amounts are required")
    started = result.attempt_started_at_ts_ms
    observed = result.response_observed_at_ts_ms
    if (
        type(started) is not int
        or type(observed) is not int
        or started < 0
        or observed < started
        or budget_commit_observed_at_ms < observed
    ):
        raise ValueError("original gateway attempt clocks are required")
    record = _canonical_json(
        {
            "version": "kairos.llm-application-call-evidence.v1",
            "capture_scope": "application-boundary-not-provider-http",
            "phase": "COMPLETED",
            "call_id": call_id,
            "attempt_record_sha256": hashlib.sha256(attempt_record).hexdigest(),
            "response": {"content_utf8": _encoded(result.content.encode("utf-8"))},
            "token_usage": asdict(result.usage),
            "provider_call": {
                "provider": result.provider,
                "model": result.model,
                "resolved_model": result.resolved_model,
                "request_id": result.request_id,
                # This legacy flag means provider input-token caching, NOT
                # whole-response reuse. Preserve its existing meaning.
                "cached": result.cached,
                "cache_scope": "PROVIDER_INPUT_TOKENS_NOT_RESPONSE_REUSE",
                "attempt_started_at_ms": started,
                "response_observed_at_ms": observed,
            },
            "budget": {
                "reservation_id": result.budget_reservation_id,
                "status": "COMMITTED",
                "reserved_microusd": reserved_microusd,
                "actual_microusd": actual_microusd,
                "commit_observed_at_ms": budget_commit_observed_at_ms,
            },
        }
    )
    if len(record) > MAX_CALL_RECORD_BYTES:
        raise ValueError("application call evidence exceeds the bounded record size")
    return record


class CreateOnlyDirectorySink:
    """Owner-only local store for one capture session; never overwrites records.

    The session directory must be new. POSIX permissions are set to 0700 for
    the directory and 0600 for files. Windows creation uses a protected DACL
    granting full control only to the object owner; failure to establish that
    DACL is fail-closed. Exact replay is a no-op; differing bytes conflict.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory).absolute()
        self._reject_reparse_ancestors(self.directory.parent)
        if self.directory.exists() or self.directory.is_symlink():
            raise FileExistsError(self.directory)
        if os.name == "nt":
            self._create_windows_directory()
        else:
            self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)
            os.chmod(self.directory, 0o700)
        self._directory_identity = self._identity(self.directory)

    @staticmethod
    def _is_reparse(info) -> bool:
        attributes = getattr(info, "st_file_attributes", 0)
        return stat.S_ISLNK(info.st_mode) or bool(attributes & 0x400)

    @classmethod
    def _reject_reparse_ancestors(cls, path: Path) -> None:
        for ancestor in (path, *path.parents):
            info = ancestor.lstat()
            if cls._is_reparse(info):
                raise ValueError("evidence path cannot contain symlink or reparse ancestors")

    @staticmethod
    def _identity(path: Path) -> tuple[int, int]:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or CreateOnlyDirectorySink._is_reparse(info):
            raise ValueError("evidence session path must remain a real directory")
        return info.st_dev, info.st_ino

    def _assert_directory(self) -> None:
        self._reject_reparse_ancestors(self.directory)
        if self._identity(self.directory) != self._directory_identity:
            raise ValueError("evidence session directory identity changed")

    def _create_windows_directory(self) -> None:
        import ctypes
        from ctypes import wintypes

        class SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", wintypes.LPVOID),
                ("bInheritHandle", wintypes.BOOL),
            ]

        descriptor = wintypes.LPVOID()
        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
        convert.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            wintypes.LPVOID,
        ]
        convert.restype = wintypes.BOOL
        # Protected DACL: only the object owner receives access. No inherited
        # grants are copied into the evidence directory.
        if not convert("D:P(A;;FA;;;OW)", 1, ctypes.byref(descriptor), None):
            raise OSError(ctypes.get_last_error(), "cannot build owner-only evidence ACL")
        try:
            attrs = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), descriptor, False)
            create = kernel.CreateDirectoryW
            create.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(SECURITY_ATTRIBUTES)]
            create.restype = wintypes.BOOL
            if not create(str(self.directory), ctypes.byref(attrs)):
                raise OSError(ctypes.get_last_error(), "cannot create protected evidence directory")
        finally:
            kernel.LocalFree(descriptor)

    def _create_windows_file(self, path: Path, record: bytes) -> None:
        import ctypes
        from ctypes import wintypes

        class SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", wintypes.LPVOID),
                ("bInheritHandle", wintypes.BOOL),
            ]

        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        descriptor = wintypes.LPVOID()
        convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
        convert.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            wintypes.LPVOID,
        ]
        convert.restype = wintypes.BOOL
        if not convert("D:P(A;;FA;;;OW)", 1, ctypes.byref(descriptor), None):
            raise OSError(ctypes.get_last_error(), "cannot build owner-only evidence file ACL")
        handle = wintypes.HANDLE(-1).value
        try:
            attrs = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), descriptor, False)
            create = kernel.CreateFileW
            create.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(SECURITY_ATTRIBUTES),
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            create.restype = wintypes.HANDLE
            handle = create(str(path), 0x40000000, 0, ctypes.byref(attrs), 1, 0x80, None)
            if handle == wintypes.HANDLE(-1).value:
                error = ctypes.get_last_error()
                if error == 80:
                    raise FileExistsError(path)
                raise OSError(error, "cannot create protected evidence file")
            buffer = ctypes.create_string_buffer(record)
            written = wintypes.DWORD()
            write = kernel.WriteFile
            write.argtypes = [
                wintypes.HANDLE,
                wintypes.LPCVOID,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                wintypes.LPVOID,
            ]
            write.restype = wintypes.BOOL
            kernel.FlushFileBuffers.argtypes = [wintypes.HANDLE]
            kernel.FlushFileBuffers.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            kernel.LocalFree.argtypes = [wintypes.HLOCAL]
            kernel.LocalFree.restype = wintypes.HLOCAL
            success = write(handle, buffer, len(record), ctypes.byref(written), None)
            if not success or written.value != len(record):
                raise OSError(ctypes.get_last_error(), "cannot write complete protected evidence file")
            if not kernel.FlushFileBuffers(handle):
                raise OSError(ctypes.get_last_error(), "cannot flush protected evidence file")
        finally:
            if handle != wintypes.HANDLE(-1).value:
                kernel.CloseHandle(handle)
            kernel.LocalFree(descriptor)

    def create_only(self, call_id: str, phase: str, record: bytes) -> None:
        if not _CALL_ID.fullmatch(call_id) or phase not in _PHASES or not isinstance(record, bytes):
            raise ValueError("normalized call ID, phase, and immutable record bytes are required")
        if len(record) > MAX_CALL_RECORD_BYTES:
            raise ValueError("application call evidence exceeds the bounded record size")
        self._assert_directory()
        path = self.directory / f"{call_id}.{phase}.json"
        try:
            if os.name == "nt":
                self._create_windows_file(path, record)
            else:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(path, flags, 0o600)
                try:
                    with os.fdopen(fd, "wb", closefd=False) as handle:
                        handle.write(record)
                        handle.flush()
                        os.fsync(fd)
                finally:
                    os.close(fd)
                os.chmod(path, 0o600)
        except FileExistsError:
            self._assert_directory()
            self._reject_reparse_ancestors(path)
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or self._is_reparse(info)
                or info.st_size > MAX_CALL_RECORD_BYTES
                or info.st_nlink != 1
            ):
                raise EvidenceConflictError("existing call evidence path is not a regular file") from None
            with path.open("rb") as handle:
                retained = handle.read(MAX_CALL_RECORD_BYTES + 1)
            if retained != record:
                raise EvidenceConflictError("call ID replay conflicts with immutable evidence") from None
            return  # Exact replay leaves original bytes and clocks unchanged.
