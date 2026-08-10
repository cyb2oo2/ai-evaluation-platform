from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path


class OutputTransactionError(RuntimeError):
    pass


class OutputTransaction:
    def __init__(self, final_dir: Path, *, force: bool) -> None:
        self.final_dir = final_dir.resolve()
        self.force = force
        self.stage_dir: Path | None = None

    def __enter__(self) -> Path:
        _validate_output_target(self.final_dir)
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.final_dir.exists():
            if not self.final_dir.is_dir():
                raise OutputTransactionError(
                    f"output path exists and is not a directory: {self.final_dir}"
                )
            if any(self.final_dir.iterdir()) and not self.force:
                raise OutputTransactionError(
                    f"output directory is not empty: {self.final_dir}; "
                    "use --force to replace it"
                )
        self.stage_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self.final_dir.name}.staging-",
                dir=self.final_dir.parent,
            )
        )
        return self.stage_dir

    def __exit__(self, exc_type, exc, traceback) -> bool:  # noqa: ANN001
        del traceback
        if self.stage_dir is None:
            return False
        if exc_type is not None:
            shutil.rmtree(self.stage_dir, ignore_errors=True)
            return False
        self._commit()
        return False

    def _commit(self) -> None:
        assert self.stage_dir is not None
        backup_dir: Path | None = None
        try:
            if self.final_dir.exists():
                backup_dir = self.final_dir.parent / (
                    f".{self.final_dir.name}.backup-{uuid.uuid4().hex}"
                )
                self.final_dir.replace(backup_dir)
            self.stage_dir.replace(self.final_dir)
        except OSError as exc:
            rollback_error = _restore_backup(self.final_dir, backup_dir)
            detail = f"; rollback also failed: {rollback_error}" if rollback_error else ""
            raise OutputTransactionError(
                f"failed to publish output directory '{self.final_dir}': {exc}{detail}"
            ) from exc
        if backup_dir is not None:
            self._remove_backup_or_rollback(backup_dir)

    def _remove_backup_or_rollback(self, backup_dir: Path) -> None:
        try:
            shutil.rmtree(backup_dir)
            return
        except OSError as cleanup_error:
            failed_dir = self.final_dir.parent / (
                f".{self.final_dir.name}.failed-{uuid.uuid4().hex}"
            )
            try:
                self.final_dir.replace(failed_dir)
                backup_dir.replace(self.final_dir)
            except OSError as rollback_error:
                raise OutputTransactionError(
                    f"published '{self.final_dir}', but backup cleanup failed: {cleanup_error}; "
                    f"rollback also failed: {rollback_error}"
                ) from cleanup_error
            try:
                shutil.rmtree(failed_dir)
            except OSError as residue_error:
                raise OutputTransactionError(
                    f"backup cleanup failed and previous output was restored; "
                    f"new output remains at '{failed_dir}': {residue_error}"
                ) from cleanup_error
            raise OutputTransactionError(
                f"backup cleanup failed; previous output was restored: {cleanup_error}"
            ) from cleanup_error


def _restore_backup(final_dir: Path, backup_dir: Path | None) -> OSError | None:
    if backup_dir is None or not backup_dir.exists() or final_dir.exists():
        return None
    try:
        backup_dir.replace(final_dir)
    except OSError as exc:
        return exc
    return None


def _validate_output_target(path: Path) -> None:
    if path == Path(path.anchor) or not path.name or path.name in {".", ".."}:
        raise OutputTransactionError(f"unsafe output directory: {path}")
