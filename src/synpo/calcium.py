from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from matplotlib.figure import Figure
from openpyxl import Workbook, load_workbook
from scipy import ndimage, stats
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay

from .measurements import filtered_measurement_result
from .preprocessing import project_cache_path
from .project import save_project


CALCIUM_PEAK_COLUMN = "spine_peak_dff"
CALCIUM_PEARSON_COLUMN = "pearson_r"
REVIEW_STATES = {"unreviewed", "confirmed", "probable", "ambiguous", "ignored"}


def calcium_analysis_directory(manifest: dict[str, object]) -> Path:
    return project_cache_path(manifest).parent / "calcium-analysis"


def calcium_dataset_path(manifest: dict[str, object], dataset_id: str) -> Path:
    return calcium_analysis_directory(manifest) / f"dataset-{dataset_id}.json.gz"


def _utc_timestamp() -> float:
    return time.time()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return cleaned or "recording"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _metric_key(label: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(label).strip()).strip("_")
    return text.casefold()


def _read_summary_workbook(path: Path) -> list[dict[str, object]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    rows: list[dict[str, object]] = []
    required_labels = {
        "spine roi",
        "parent dendrite segment",
        "spine peak dff/f0",
        "pearson r",
    }
    try:
        for sheet in workbook.worksheets:
            header: list[str] | None = None
            for values in sheet.iter_rows(values_only=True):
                normalized = [str(value).strip() if value is not None else "" for value in values]
                if header is None:
                    if required_labels.issubset(
                        {value.casefold() for value in normalized if value}
                    ):
                        header = normalized
                    continue
                source = {
                    label: values[index] if index < len(values) else None
                    for index, label in enumerate(header)
                    if label
                }
                spine_roi = source.get("Spine ROI")
                if spine_roi is None or not str(spine_roi).strip():
                    continue
                parent = source.get("Parent dendrite segment")
                row = {
                    "recording_id": sheet.title,
                    "spine_roi": str(spine_roi).strip(),
                    "parent_segment": "" if parent is None else str(parent).strip(),
                    CALCIUM_PEAK_COLUMN: source.get("Spine peak dFF/F0"),
                    CALCIUM_PEARSON_COLUMN: source.get("Pearson r"),
                }
                row.update(
                    {
                        _metric_key(label): value
                        for label, value in source.items()
                        if value is not None and _metric_key(label)
                    }
                )
                rows.append(row)
    finally:
        workbook.close()
    return rows


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None and float(number).is_integer() else None


def _boolean(value: object, default: bool = False) -> bool:
    text = str(value).strip().casefold()
    if text in {"true", "yes", "1", "include", "included"}:
        return True
    if text in {"false", "no", "0", "exclude", "excluded"}:
        return False
    return default


def _numeric_metrics(row: dict[str, object]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for key, value in row.items():
        number = _optional_float(value)
        if number is None:
            continue
        integer = _optional_int(value)
        metrics[str(key)] = integer if integer is not None else number
    return metrics


def read_imagej_roi(data: bytes) -> dict[str, object]:
    """Read polygon/freehand coordinates from an ImageJ .roi member."""

    if len(data) < 64 or data[:4] != b"Iout":
        raise ValueError("Unsupported or truncated ImageJ ROI.")
    roi_type = int(data[6])
    if roi_type not in {0, 7, 8}:
        raise ValueError(f"Unsupported ImageJ ROI type {roi_type}; expected polygon/freehand.")
    top = int.from_bytes(data[8:10], "big", signed=True)
    left = int.from_bytes(data[10:12], "big", signed=True)
    bottom = int.from_bytes(data[12:14], "big", signed=True)
    right = int.from_bytes(data[14:16], "big", signed=True)
    count = int.from_bytes(data[16:18], "big", signed=False)
    if count <= 0 or 64 + 4 * count > len(data):
        raise ValueError("ImageJ ROI has an invalid coordinate count.")
    options = int.from_bytes(data[50:52], "big", signed=False)
    integer_x = np.frombuffer(data, dtype=">i2", count=count, offset=64).astype(np.float64)
    integer_y = np.frombuffer(
        data, dtype=">i2", count=count, offset=64 + 2 * count
    ).astype(np.float64)
    x = integer_x + left
    y = integer_y + top
    float_offset = 64 + 4 * count
    if options & 128 and float_offset + 8 * count <= len(data):
        x = np.frombuffer(
            data, dtype=">f4", count=count, offset=float_offset
        ).astype(np.float64)
        y = np.frombuffer(
            data, dtype=">f4", count=count, offset=float_offset + 4 * count
        ).astype(np.float64)
    points = [[float(x_value), float(y_value)] for x_value, y_value in zip(x, y)]
    return {
        "type": roi_type,
        "bounds_xyxy": [left, top, right, bottom],
        "points_xy": points,
        "center_xy": [float(np.mean(x)), float(np.mean(y))],
    }


def _read_roi_zip(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    values: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.casefold().endswith(".roi"):
                continue
            name = Path(member.filename).stem
            try:
                values[name] = read_imagej_roi(archive.read(member))
            except ValueError as exc:
                raise ValueError(f"Cannot read ROI {member.filename!r} in {path.name}: {exc}") from exc
    return values


def _manifest_centers(path: Path) -> dict[str, tuple[float, float]]:
    if not path.is_file():
        return {}
    values: dict[str, tuple[float, float]] = {}
    for row in _read_csv(path):
        x = _optional_float(row.get("center_x"))
        y = _optional_float(row.get("center_y"))
        if x is not None and y is not None:
            values[str(row.get("roi_name", ""))] = (x, y)
    return values


def _recording_directory(root: Path, recording_id: str) -> Path:
    direct = root / recording_id
    if direct.is_dir():
        return direct
    folded = recording_id.casefold()
    candidates = [path for path in root.iterdir() if path.is_dir() and path.name.casefold() == folded]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"The results folder is missing recording directory {recording_id!r}.")


def _dataset_record(manifest: dict[str, object], dataset_id: str) -> dict[str, object]:
    analysis = manifest.get("calcium_analysis", {})
    for record in analysis.get("datasets", []):
        if str(record.get("dataset_id")) == str(dataset_id):
            return record
    raise ValueError("The selected calcium dataset is not registered in this project.")


def _calcium_clustering_signature(dataset: dict[str, object]) -> str:
    payload = {
        "calcium_values": sorted(
            (
                str(spine.get("calcium_uid", "")),
                spine.get("calcium_spine_peak_dff"),
                spine.get("calcium_pearson_r"),
                str(spine.get("synpo_field_coverage", "unknown")),
            )
            for spine in _calcium_spines(dataset)
        ),
        "matches": sorted(
            (
                str(row.get("calcium_uid", "")),
                row.get("synpo_specimen_index"),
                row.get("synpo_spine_id"),
                str(row.get("review_state", "unreviewed")),
            )
            for row in dataset.get("matches", [])
            if str(row.get("review_state", "unreviewed"))
            in {"confirmed", "probable", "ambiguous"}
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_calcium_dataset(
    manifest: dict[str, object], dataset_id: str | None = None
) -> dict[str, object]:
    analysis = manifest.get("calcium_analysis", {})
    selected = str(dataset_id or analysis.get("active_dataset_id") or "")
    if not selected:
        raise ValueError("Import a calcium-imaging dataset first.")
    _dataset_record(manifest, selected)
    path = calcium_dataset_path(manifest, selected)
    if not path.is_file():
        raise ValueError(
            "The saved calcium dataset cache is unavailable. Restore the full "
            f".synpo-cache or a full transfer ZIP. Expected: {path}"
        )
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        dataset = json.load(stream)
    update_calcium_field_coverage_from_manifest(manifest, dataset)
    return dataset


def save_calcium_dataset(
    manifest: dict[str, object],
    project_path: str | Path,
    dataset: dict[str, object],
) -> Path:
    dataset_id = str(dataset["dataset_id"])
    dataset["updated_at"] = _utc_timestamp()
    path = calcium_dataset_path(manifest, dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(dataset, stream, separators=(",", ":"))
    os.replace(temporary, path)
    record = _dataset_record(manifest, dataset_id)
    clustering_signature = _calcium_clustering_signature(dataset)
    previous_signature = str(record.get("clustering_signature", ""))
    if previous_signature and previous_signature != clustering_signature:
        for run in manifest.get("morphology_analysis", {}).get("runs", []):
            settings = dict(run.get("settings", {}))
            raw_selected_ids = settings.get("calcium_dataset_ids", [])
            selected_ids = (
                {str(value) for value in raw_selected_ids}
                if isinstance(raw_selected_ids, (list, tuple, set))
                else ({str(raw_selected_ids)} if raw_selected_ids else set())
            )
            legacy_id = str(settings.get("calcium_dataset_id", ""))
            if dataset_id == legacy_id or dataset_id in selected_ids:
                run["stale"] = True
    record.update(
        {
            "name": dataset.get("name", record.get("name", "Calcium dataset")),
            "updated_at": dataset["updated_at"],
            "registration_count": len(dataset.get("registrations", [])),
            "match_count": sum(
                bool(row.get("synpo_spine_id")) for row in dataset.get("matches", [])
            ),
            "reviewed_match_count": sum(
                str(row.get("review_state")) in {"confirmed", "probable", "ambiguous", "ignored"}
                for row in dataset.get("matches", [])
            ),
            "clustering_signature": clustering_signature,
        }
    )
    manifest.setdefault("calcium_analysis", {})["active_dataset_id"] = dataset_id
    save_project(project_path, manifest)
    return path


def invalidate_calcium_matches_for_specimen(
    manifest: dict[str, object],
    project_path: str | Path,
    specimen_index: int,
    *,
    reason: str = "Synpo preprocessing or detection was rerun",
) -> dict[str, int]:
    """Require renewed match review while retaining field registrations.

    Reprocessing does not change the raw-image coordinate system, so saved
    calcium/Synpo landmark pairs remain valid.  Detected spine label IDs can
    change, however.  Keep each former assignment as a visible proposal, but
    return it to the unreviewed state so it cannot silently enter a calcium or
    clustering analysis until it has been checked against the new detection.
    """

    target = int(specimen_index)
    invalidated = 0
    updated_datasets = 0
    preserved_registrations = 0
    for record in manifest.get("calcium_analysis", {}).get("datasets", []):
        dataset_id = str(record.get("dataset_id", ""))
        if not dataset_id:
            continue
        try:
            dataset = load_calcium_dataset(manifest, dataset_id)
        except (OSError, ValueError):
            # A missing externally transferred calcium cache must not prevent
            # the core Synpo pipeline from being rerun.
            continue
        registrations = [
            row
            for row in dataset.get("registrations", [])
            if int(row.get("synpo_specimen_index", -1)) == target
        ]
        preserved_registrations += len(registrations)
        changed = False
        for match in dataset.get("matches", []):
            if match.get("synpo_specimen_index") is None or int(
                match.get("synpo_specimen_index", -1)
            ) != target:
                continue
            if bool(match.get("synpo_result_stale", False)):
                continue
            match["review_state_before_synpo_change"] = str(
                match.get("review_state", "unreviewed")
            )
            invalidated += 1
            match.update(
                {
                    "review_state": "unreviewed",
                    "synpo_result_stale": True,
                    "synpo_result_stale_reason": str(reason),
                    "synpo_result_stale_at": _utc_timestamp(),
                    "updated_at": _utc_timestamp(),
                }
            )
            changed = True
        if changed:
            save_calcium_dataset(manifest, project_path, dataset)
            updated_datasets += 1
    return {
        "invalidated_matches": invalidated,
        "updated_datasets": updated_datasets,
        "preserved_registrations": preserved_registrations,
    }


def import_calcium_results(
    manifest: dict[str, object],
    project_path: str | Path,
    source_root: str | Path,
    *,
    name: str | None = None,
    embed_reference_images: bool = True,
) -> dict[str, object]:
    """Import the canonical outputs of the calcium pipeline without altering them."""

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Calcium results folder does not exist: {root}")
    metrics_path = root / "experiment_spine_metrics.csv"
    metrics_workbook = root / "experiment_spine_summary.xlsx"
    correlations_path = root / "experiment_correlations.csv"
    if metrics_path.is_file():
        metric_rows = _read_csv(metrics_path)
    elif metrics_workbook.is_file():
        metrics_path = metrics_workbook
        metric_rows = _read_summary_workbook(metrics_path)
    else:
        raise ValueError(
            "The selected folder contains neither experiment_spine_metrics.csv "
            "nor experiment_spine_summary.xlsx."
        )
    if not metric_rows:
        raise ValueError("The calcium spine-metrics table is empty.")
    required = {"recording_id", "spine_roi", "parent_segment", CALCIUM_PEAK_COLUMN, CALCIUM_PEARSON_COLUMN}
    missing = required.difference(metric_rows[0])
    if missing:
        raise ValueError(
            "Calcium spine-metrics table is missing required columns: "
            + ", ".join(sorted(missing))
        )
    correlations = {
        (str(row.get("recording_id", "")), str(row.get("spine_roi", ""))): row
        for row in (_read_csv(correlations_path) if correlations_path.is_file() else [])
    }
    rows_by_recording: dict[str, list[dict[str, str]]] = {}
    for row in metric_rows:
        recording_id = str(row.get("recording_id", "")).strip()
        spine_roi = str(row.get("spine_roi", "")).strip()
        if recording_id and spine_roi:
            rows_by_recording.setdefault(recording_id, []).append(row)
    if not rows_by_recording:
        raise ValueError("The calcium table contains no identified spine rows.")

    dataset_id = uuid.uuid4().hex
    analysis_root = calcium_analysis_directory(manifest)
    image_root = analysis_root / "images" / dataset_id
    copied_images: list[Path] = []
    recordings: list[dict[str, object]] = []
    try:
        for recording_id, rows in sorted(rows_by_recording.items()):
            directory = _recording_directory(root, recording_id)
            spine_rois = _read_roi_zip(
                directory / "segmentation" / "spines" / "included_spines.zip"
            )
            dendrite_zip = directory / "segmentation" / "dendrites_corrected" / "included_segments.zip"
            if not dendrite_zip.is_file():
                dendrite_zip = directory / "segmentation" / "dendrites" / "included_segments.zip"
            dendrite_rois = _read_roi_zip(dendrite_zip)
            spine_centers = _manifest_centers(
                directory / "segmentation" / "spines" / "spine_manifest.csv"
            )
            dendrite_manifest = directory / "segmentation" / "dendrites_corrected" / "segment_manifest.csv"
            if not dendrite_manifest.is_file():
                dendrite_manifest = directory / "segmentation" / "dendrites" / "segment_manifest.csv"
            dendrite_centers = _manifest_centers(dendrite_manifest)
            reference = directory / "motion_corrected.png"
            image_member: str | None = None
            external_image = str(reference) if reference.is_file() else None
            if reference.is_file() and embed_reference_images:
                image_root.mkdir(parents=True, exist_ok=True)
                destination = image_root / f"{_safe_name(recording_id)}.png"
                shutil.copy2(reference, destination)
                copied_images.append(destination)
                image_member = destination.relative_to(analysis_root).as_posix()

            spine_values: list[dict[str, object]] = []
            for row in rows:
                spine_name = str(row["spine_roi"])
                parent_name = str(row.get("parent_segment", ""))
                roi = spine_rois.get(spine_name, {})
                center = roi.get("center_xy")
                if center is None and spine_name in spine_centers:
                    center = list(spine_centers[spine_name])
                if center is None:
                    raise ValueError(
                        f"Recording {recording_id}, spine {spine_name} has no ROI coordinates."
                    )
                merged_metrics = _numeric_metrics(row)
                correlation_row = correlations.get((recording_id, spine_name), {})
                merged_metrics.update(_numeric_metrics(correlation_row))
                spine_values.append(
                    {
                        "calcium_uid": f"{recording_id}/{spine_name}",
                        "recording_id": recording_id,
                        "spine_roi": spine_name,
                        "parent_segment": parent_name,
                        "biological_dendrite_id": str(row.get("biological_dendrite_id", "")),
                        "proposal_source": str(row.get("proposal_source", "")),
                        "source_included": _boolean(row.get("signal_review_include"), True),
                        "center_xy": [float(center[0]), float(center[1])],
                        "roi": roi,
                        "parent_center_xy": list(
                            dendrite_rois.get(parent_name, {}).get(
                                "center_xy", dendrite_centers.get(parent_name, center)
                            )
                        ),
                        "metrics": merged_metrics,
                        "calcium_spine_peak_dff": _optional_float(row.get(CALCIUM_PEAK_COLUMN)),
                        "calcium_pearson_r": _optional_float(row.get(CALCIUM_PEARSON_COLUMN)),
                    }
                )
            recordings.append(
                {
                    "recording_id": recording_id,
                    "source_directory": str(directory),
                    "reference_image_member": image_member,
                    "reference_image_path": external_image,
                    "spines": spine_values,
                    "dendrite_segments": {
                        key: {**value, "center_xy": value.get("center_xy", list(dendrite_centers.get(key, (0.0, 0.0))))}
                        for key, value in dendrite_rois.items()
                    },
                }
            )
    except Exception:
        for copied in copied_images:
            copied.unlink(missing_ok=True)
        if image_root.is_dir() and not any(image_root.iterdir()):
            image_root.rmdir()
        raise

    dataset = {
        "format": "synpo-calcium-matching",
        "format_version": 1,
        "dataset_id": dataset_id,
        "name": str(name or root.name),
        "experiment_label": root.name,
        "source_root": str(root),
        "source_table": str(metrics_path),
        "source_fingerprint": _source_fingerprint(metrics_path),
        "created_at": _utc_timestamp(),
        "updated_at": _utc_timestamp(),
        "embedded_reference_images": bool(embed_reference_images),
        "recordings": recordings,
        "registrations": [],
        "matches": [],
        "analysis_runs": [],
        "active_analysis_run_id": None,
        "analysis_settings": {
            "accepted_match_states": ["confirmed"],
            "calcium_metric": "calcium_spine_peak_dff",
            "x_metric": "protein_puncta_total_volume_um3",
            "show_regression": True,
            "show_confidence_interval": True,
            "point_alpha": 0.72,
            "log_x": False,
            "log_y": False,
        },
    }
    analysis = manifest.setdefault(
        "calcium_analysis", {"datasets": [], "active_dataset_id": None}
    )
    record = {
        "dataset_id": dataset_id,
        "name": dataset["name"],
        "experiment_label": dataset["experiment_label"],
        "source_root": str(root),
        "source_fingerprint": dataset["source_fingerprint"],
        "created_at": dataset["created_at"],
        "updated_at": dataset["updated_at"],
        "recording_count": len(recordings),
        "spine_count": sum(len(row["spines"]) for row in recordings),
        "embedded_reference_images": bool(embed_reference_images),
        "registration_count": 0,
        "match_count": 0,
        "reviewed_match_count": 0,
        "clustering_signature": _calcium_clustering_signature(dataset),
    }
    analysis.setdefault("datasets", []).append(record)
    analysis["active_dataset_id"] = dataset_id
    path = calcium_dataset_path(manifest, dataset_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
            json.dump(dataset, stream, separators=(",", ":"))
        os.replace(temporary, path)
        save_project(project_path, manifest)
    except Exception:
        analysis["datasets"].remove(record)
        if analysis.get("active_dataset_id") == dataset_id:
            analysis["active_dataset_id"] = None
        path.unlink(missing_ok=True)
        raise
    return dataset


def upsert_calcium_analysis_run(
    dataset: dict[str, object],
    name: str,
    settings: dict[str, object],
) -> dict[str, object]:
    clean_name = str(name).strip()
    if not clean_name:
        raise ValueError("Enter a name for the calcium analysis run.")
    runs = dataset.setdefault("analysis_runs", [])
    existing = next(
        (row for row in runs if str(row.get("name", "")) == clean_name),
        None,
    )
    created_at = (
        float(existing.get("created_at", _utc_timestamp()))
        if existing is not None
        else _utc_timestamp()
    )
    record = existing if existing is not None else {}
    record.update(
        {
            "run_id": str(record.get("run_id") or uuid.uuid4().hex),
            "name": clean_name,
            "created_at": created_at,
            "updated_at": _utc_timestamp(),
            "settings": dict(settings),
        }
    )
    if existing is None:
        runs.append(record)
    dataset["active_analysis_run_id"] = record["run_id"]
    dataset["analysis_settings"] = dict(settings)
    return record


def calcium_reference_image_path(
    manifest: dict[str, object], dataset: dict[str, object], recording_id: str
) -> Path:
    recording = next(
        (row for row in dataset.get("recordings", []) if str(row.get("recording_id")) == recording_id),
        None,
    )
    if recording is None:
        raise ValueError(f"Unknown calcium recording {recording_id!r}.")
    member = str(recording.get("reference_image_member") or "")
    if member:
        cached = calcium_analysis_directory(manifest) / member
        if cached.is_file():
            return cached
    external = Path(str(recording.get("reference_image_path") or "")).expanduser()
    if external.is_file():
        return external.resolve()
    raise ValueError(
        f"The reference image for calcium recording {recording_id!r} is unavailable."
    )


def relink_calcium_source(
    manifest: dict[str, object],
    project_path: str | Path,
    dataset_id: str,
    source_root: str | Path,
) -> dict[str, object]:
    root = Path(source_root).expanduser().resolve()
    dataset = load_calcium_dataset(manifest, dataset_id)
    expected = dict(dataset.get("source_fingerprint", {}))
    expected_name = str(expected.get("name", "experiment_spine_metrics.csv"))
    metrics = root / expected_name
    if not metrics.is_file():
        alternatives = [
            root / "experiment_spine_metrics.csv",
            root / "experiment_spine_summary.xlsx",
        ]
        metrics = next((path for path in alternatives if path.is_file()), metrics)
    if not metrics.is_file():
        raise ValueError(
            "The selected folder has no calcium spine metrics CSV or summary workbook."
        )
    actual = _source_fingerprint(metrics)
    if expected.get("sha256") and expected.get("sha256") != actual.get("sha256"):
        raise ValueError(
            "The selected calcium results table does not match the imported dataset checksum."
        )
    dataset["source_root"] = str(root)
    dataset["source_table"] = str(metrics)
    for recording in dataset.get("recordings", []):
        directory = _recording_directory(root, str(recording["recording_id"]))
        recording["source_directory"] = str(directory)
        reference = directory / "motion_corrected.png"
        recording["reference_image_path"] = str(reference) if reference.is_file() else None
    record = _dataset_record(manifest, dataset_id)
    record["source_root"] = str(root)
    save_calcium_dataset(manifest, project_path, dataset)
    return dataset


def _minimum_landmarks(model: str) -> int:
    return {"similarity": 2, "affine": 3, "piecewise": 4}.get(model, 10_000)


def _global_transform(
    source_xy: np.ndarray, target_xy: np.ndarray, model: str
) -> tuple[np.ndarray, np.ndarray]:
    if model == "similarity":
        source_center = np.mean(source_xy, axis=0)
        target_center = np.mean(target_xy, axis=0)
        centered_source = source_xy - source_center
        centered_target = target_xy - target_center
        rotation, _scale_sum = orthogonal_procrustes(centered_source, centered_target)
        denominator = float(np.sum(centered_source * centered_source))
        if denominator <= 1e-12:
            raise ValueError("Similarity landmarks do not span a measurable distance.")
        scale = float(np.sum((centered_source @ rotation) * centered_target) / denominator)
        linear = rotation * scale
        offset = target_center - source_center @ linear
        return linear, offset
    design = np.column_stack([source_xy, np.ones(len(source_xy), dtype=np.float64)])
    coefficients, _residuals, rank, _values = np.linalg.lstsq(design, target_xy, rcond=None)
    if rank < 3:
        raise ValueError("Affine landmarks are collinear or duplicated.")
    return coefficients[:2, :], coefficients[2, :]


def transform_points(
    registration: dict[str, object], points_xy: Iterable[Iterable[float]]
) -> np.ndarray:
    points = np.asarray(list(points_xy), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Registration points must be X/Y coordinate pairs.")
    model = str(registration.get("model", "affine"))
    source = np.asarray(registration.get("calcium_points_xy", []), dtype=np.float64)
    target = np.asarray(registration.get("synpo_points_xy", []), dtype=np.float64)
    if len(source) < _minimum_landmarks(model) or source.shape != target.shape:
        raise ValueError("The registration does not contain enough paired landmarks.")
    base_model = "similarity" if model == "similarity" else "affine"
    linear, offset = _global_transform(source, target, base_model)
    transformed = points @ linear + offset
    if model != "piecewise":
        return transformed
    triangulation = Delaunay(source)
    simplex_ids = triangulation.find_simplex(points)
    for point_index, simplex_id in enumerate(simplex_ids):
        if simplex_id < 0:
            continue
        vertices = triangulation.simplices[int(simplex_id)]
        local_linear, local_offset = _global_transform(
            source[vertices], target[vertices], "affine"
        )
        transformed[point_index] = points[point_index] @ local_linear + local_offset
    return transformed


def update_calcium_field_coverage(
    dataset: dict[str, object], recording_id: str | None = None
) -> dict[str, int]:
    """Classify calcium ROIs against every registered Synpo field.

    A spine is outside only when every registration for its calcium recording
    has a known image shape and its transformed center is outside all of them.
    Missing shapes or invalid transforms remain ``unknown`` so the review UI
    never hides a potentially valid ROI merely because metadata is incomplete.
    """

    registrations_by_recording: dict[str, list[dict[str, object]]] = {}
    for registration in dataset.get("registrations", []):
        registrations_by_recording.setdefault(
            str(registration.get("recording_id", "")), []
        ).append(registration)
    counts = {
        "inside": 0,
        "outside": 0,
        "unknown": 0,
        "unregistered": 0,
        "changed": 0,
    }
    for recording in dataset.get("recordings", []):
        current_recording_id = str(recording.get("recording_id", ""))
        if recording_id is not None and current_recording_id != str(recording_id):
            continue
        registrations = registrations_by_recording.get(current_recording_id, [])
        for spine in recording.get("spines", []):
            inside_registration_ids: list[str] = []
            evaluated = 0
            for registration in registrations:
                raw_shape = registration.get("synpo_shape_yx", [])
                if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
                    continue
                height, width = (int(raw_shape[0]), int(raw_shape[1]))
                if height <= 0 or width <= 0:
                    continue
                try:
                    center = transform_points(
                        registration, [spine.get("center_xy", (float("nan"),) * 2)]
                    )[0]
                except (TypeError, ValueError):
                    continue
                if not np.all(np.isfinite(center)):
                    continue
                evaluated += 1
                x, y = float(center[0]), float(center[1])
                if 0.0 <= x < width and 0.0 <= y < height:
                    inside_registration_ids.append(
                        str(registration.get("registration_id", ""))
                    )
            if inside_registration_ids:
                state = "inside"
            elif registrations and evaluated == len(registrations):
                state = "outside"
            elif registrations:
                state = "unknown"
            else:
                state = "unregistered"
            updated = {
                "synpo_field_coverage": state,
                "synpo_inside_registration_ids": inside_registration_ids,
                "synpo_registered_field_count": len(registrations),
                "synpo_evaluated_field_count": evaluated,
            }
            if any(spine.get(key) != value for key, value in updated.items()):
                counts["changed"] += 1
            spine.update(updated)
            counts[state] += 1
    return counts


def calcium_spine_in_review_scope(spine: dict[str, object]) -> bool:
    """Return false only for ROIs proven outside every registered Synpo field."""

    return str(spine.get("synpo_field_coverage", "unknown")) != "outside"


def update_calcium_field_coverage_from_manifest(
    manifest: dict[str, object], dataset: dict[str, object]
) -> dict[str, int]:
    """Backfill field sizes from project metadata, then classify calcium ROIs."""

    specimens = manifest.get("specimens", [])
    for registration in dataset.get("registrations", []):
        specimen_index = int(registration.get("synpo_specimen_index", -1))
        if not 0 <= specimen_index < len(specimens):
            continue
        for channel in specimens[specimen_index].get("channels", {}).values():
            metadata = channel.get("metadata", {})
            raw_shape = (
                metadata.get("shape", []) if isinstance(metadata, dict) else []
            )
            if isinstance(raw_shape, (list, tuple)) and len(raw_shape) >= 2:
                registration["synpo_shape_yx"] = [
                    int(raw_shape[-2]),
                    int(raw_shape[-1]),
                ]
                break
    return update_calcium_field_coverage(dataset)


def upsert_registration(
    dataset: dict[str, object],
    recording_id: str,
    synpo_specimen_index: int,
    model: str,
    calcium_points_xy: Iterable[Iterable[float]],
    synpo_points_xy: Iterable[Iterable[float]],
    synpo_shape_yx: Iterable[int] | None = None,
) -> dict[str, object]:
    if model not in {"similarity", "affine", "piecewise"}:
        raise ValueError("Registration model must be similarity, affine, or piecewise.")
    source = np.asarray(list(calcium_points_xy), dtype=np.float64)
    target = np.asarray(list(synpo_points_xy), dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Calcium and Synpo landmarks must be paired X/Y coordinates.")
    minimum = _minimum_landmarks(model)
    if len(source) < minimum:
        raise ValueError(f"{model.title()} registration requires at least {minimum} landmark pairs.")
    draft = {
        "model": model,
        "calcium_points_xy": source.tolist(),
        "synpo_points_xy": target.tolist(),
    }
    fitted = transform_points(draft, source)
    residuals = np.linalg.norm(fitted - target, axis=1)
    normalized_shape: tuple[int, int] | None = None
    if synpo_shape_yx is not None:
        shape = tuple(int(value) for value in synpo_shape_yx)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("The Synpo registration image shape must be positive Y/X values.")
        normalized_shape = (shape[0], shape[1])
    registrations = dataset.setdefault("registrations", [])
    existing = next(
        (
            row
            for row in registrations
            if str(row.get("recording_id")) == str(recording_id)
            and int(row.get("synpo_specimen_index", -1)) == int(synpo_specimen_index)
        ),
        None,
    )
    registration = existing if existing is not None else {}
    registration.update(
        {
            "registration_id": str(registration.get("registration_id") or uuid.uuid4().hex),
            "recording_id": str(recording_id),
            "synpo_specimen_index": int(synpo_specimen_index),
            **draft,
            "landmark_count": len(source),
            "rms_error_px": float(np.sqrt(np.mean(residuals**2))),
            "maximum_error_px": float(np.max(residuals)),
            "updated_at": _utc_timestamp(),
        }
    )
    if normalized_shape is not None:
        registration["synpo_shape_yx"] = list(normalized_shape)
    if existing is None:
        registrations.append(registration)
    update_calcium_field_coverage(dataset, str(recording_id))
    return registration


def remove_registration(
    dataset: dict[str, object], recording_id: str, synpo_specimen_index: int
) -> bool:
    registrations = dataset.setdefault("registrations", [])
    retained = [
        row
        for row in registrations
        if not (
            str(row.get("recording_id")) == str(recording_id)
            and int(row.get("synpo_specimen_index", -1)) == int(synpo_specimen_index)
        )
    ]
    changed = len(retained) != len(registrations)
    registrations[:] = retained
    if changed:
        update_calcium_field_coverage(dataset, str(recording_id))
    return changed


def _calcium_spines(dataset: dict[str, object]) -> list[dict[str, object]]:
    return [
        spine
        for recording in dataset.get("recordings", [])
        for spine in recording.get("spines", [])
    ]


def _find_calcium_spine(dataset: dict[str, object], calcium_uid: str) -> dict[str, object]:
    for spine in _calcium_spines(dataset):
        if str(spine.get("calcium_uid")) == str(calcium_uid):
            return spine
    raise ValueError(f"Unknown calcium spine {calcium_uid!r}.")


def synpo_matching_scene(
    manifest: dict[str, object], specimen_index: int
) -> dict[str, object]:
    """Build the 2D intensity/label scene used for registration and matching."""

    import zarr

    from .measurements import _mask_sources

    editable, _detection, _corrected, _signature = _mask_sources(manifest, specimen_index)
    spine_data = editable["spine_labels"]
    dendrite_data = editable["dendrite_labels"]
    shape = tuple(int(value) for value in spine_data.shape)
    spine_projection = np.zeros(shape[-2:], dtype=np.uint32)
    dendrite_projection = np.zeros(shape[-2:], dtype=np.uint32)
    spine_counts = np.zeros(1, dtype=np.int64)
    spine_x_sums = np.zeros(1, dtype=np.float64)
    spine_y_sums = np.zeros(1, dtype=np.float64)
    for z_index in range(shape[0]):
        spine_slice = np.asarray(spine_data[z_index], dtype=np.uint32)
        np.maximum(spine_projection, spine_slice, out=spine_projection)
        flat = spine_slice.ravel()
        positions = np.flatnonzero(flat)
        if len(positions):
            labels = flat[positions].astype(np.int64, copy=False)
            required = int(np.max(labels)) + 1
            if required > len(spine_counts):
                growth = required - len(spine_counts)
                spine_counts = np.pad(spine_counts, (0, growth))
                spine_x_sums = np.pad(spine_x_sums, (0, growth))
                spine_y_sums = np.pad(spine_y_sums, (0, growth))
            spine_counts[:required] += np.bincount(labels, minlength=required)
            spine_x_sums[:required] += np.bincount(
                labels,
                weights=positions % shape[-1],
                minlength=required,
            )
            spine_y_sums[:required] += np.bincount(
                labels,
                weights=positions // shape[-1],
                minlength=required,
            )
        np.maximum(
            dendrite_projection,
            np.asarray(dendrite_data[z_index], dtype=np.uint32),
            out=dendrite_projection,
        )
    dendrite_channel = next(
        channel
        for channel, role in manifest["channel_roles"].items()
        if role == "dendrite_spines"
    )
    key = manifest["specimens"][specimen_index]["checkpoints"]["preprocessing"]["channels"][dendrite_channel]["dataset_key"]
    root = zarr.open_group(str(project_cache_path(manifest)), mode="r")
    raw_data = root[key]
    raw_projection = np.zeros(shape[-2:], dtype=np.float32)
    for z_index in range(shape[0]):
        np.maximum(raw_projection, np.asarray(raw_data[z_index], dtype=np.float32), out=raw_projection)

    result, _audit = filtered_measurement_result(manifest, specimen_index)
    rows_by_id = {int(row["spine_id"]): row for row in result.get("spine_rows", [])}
    identifiers = [
        spine_id
        for spine_id in range(1, len(spine_counts))
        if spine_counts[spine_id] > 0
    ]
    objects: list[dict[str, object]] = []
    for spine_id in identifiers:
        row = rows_by_id.get(spine_id)
        if row is None:
            continue
        objects.append(
            {
                "spine_id": spine_id,
                "center_xy": [
                    float(spine_x_sums[spine_id] / spine_counts[spine_id]),
                    float(spine_y_sums[spine_id] / spine_counts[spine_id]),
                ],
                "dendrite_id": int(row.get("dendrite_id") or 0),
                "spine_valid": bool(row.get("spine_valid", True)),
                "volume_filter_excluded": bool(row.get("volume_filter_excluded", False)),
            }
        )
    return {
        "raw_projection": raw_projection,
        "spine_projection": spine_projection,
        "dendrite_projection": dendrite_projection,
        "spines": objects,
    }


def _reviewed_match(row: dict[str, object]) -> bool:
    return str(row.get("review_state", "unreviewed")) != "unreviewed"


def automatic_match_spines(
    manifest: dict[str, object],
    dataset: dict[str, object],
    *,
    maximum_distance_px: float = 30.0,
    scene_loader: Callable[[dict[str, object], int], dict[str, object]] = synpo_matching_scene,
) -> dict[str, int]:
    if maximum_distance_px <= 0:
        raise ValueError("Maximum automatic-match distance must be positive.")
    preserved = [dict(row) for row in dataset.get("matches", []) if _reviewed_match(row)]
    used_calcium = {str(row.get("calcium_uid")) for row in preserved}
    used_targets = {
        (int(row["synpo_specimen_index"]), int(row["synpo_spine_id"]))
        for row in preserved
        if row.get("synpo_specimen_index") is not None and row.get("synpo_spine_id") is not None
    }
    proposed: list[dict[str, object]] = []
    scene_cache: dict[int, dict[str, object]] = {}
    registrations_by_recording: dict[str, list[dict[str, object]]] = {}
    for registration in dataset.get("registrations", []):
        registrations_by_recording.setdefault(str(registration["recording_id"]), []).append(registration)

    for recording in dataset.get("recordings", []):
        recording_id = str(recording["recording_id"])
        registrations = registrations_by_recording.get(recording_id, [])
        for registration in registrations:
            specimen_index = int(registration["synpo_specimen_index"])
            scene = scene_cache.get(specimen_index)
            if scene is None:
                scene = scene_loader(manifest, specimen_index)
                scene_cache[specimen_index] = scene
            field_projection = scene.get(
                "spine_projection", scene.get("dendrite_projection")
            )
            if field_projection is None:
                raise ValueError(
                    "The Synpo matching scene does not contain an image-sized projection."
                )
            registration["synpo_shape_yx"] = list(
                np.asarray(field_projection).shape[-2:]
            )
        update_calcium_field_coverage(dataset, recording_id)
        calcium_rows = [
            row
            for row in recording.get("spines", [])
            if bool(row.get("source_included", True))
            and calcium_spine_in_review_scope(row)
            and str(row.get("calcium_uid")) not in used_calcium
        ]
        if not registrations or not calcium_rows:
            continue
        targets: list[tuple[dict[str, object], dict[str, object]]] = []
        for registration in registrations:
            specimen_index = int(registration["synpo_specimen_index"])
            scene = scene_cache[specimen_index]
            for target in scene.get("spines", []):
                key = (specimen_index, int(target["spine_id"]))
                if key in used_targets or not bool(target.get("spine_valid", True)) or bool(
                    target.get("volume_filter_excluded", False)
                ):
                    continue
                targets.append((registration, target))
        unique_targets: dict[tuple[int, int], tuple[dict[str, object], dict[str, object]]] = {}
        for registration, target in targets:
            unique_targets[
                (int(registration["synpo_specimen_index"]), int(target["spine_id"]))
            ] = (registration, target)
        target_values = list(unique_targets.values())
        if not target_values:
            continue
        huge = 1e9
        costs = np.full((len(calcium_rows), len(target_values)), huge, dtype=np.float64)
        geometric_distances = np.full_like(costs, huge)
        dendrite_distances = np.full_like(costs, huge)
        transformed_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        distance_cache: dict[tuple[int, int], np.ndarray] = {}
        for target_index, (registration, target) in enumerate(target_values):
            registration_id = str(registration["registration_id"])
            specimen_index = int(registration["synpo_specimen_index"])
            scene = scene_cache[specimen_index]
            parent_id = int(target.get("dendrite_id") or 0)
            if parent_id and (specimen_index, parent_id) not in distance_cache:
                distance_cache[(specimen_index, parent_id)] = ndimage.distance_transform_edt(
                    np.asarray(scene["dendrite_projection"]) != parent_id
                )
            for calcium_index, calcium in enumerate(calcium_rows):
                cache_key = (str(calcium["calcium_uid"]), registration_id)
                if cache_key not in transformed_cache:
                    points = transform_points(
                        registration,
                        [calcium["center_xy"], calcium.get("parent_center_xy", calcium["center_xy"])],
                    )
                    transformed_cache[cache_key] = (points[0], points[1])
                transformed_spine, transformed_parent = transformed_cache[cache_key]
                target_center = np.asarray(target["center_xy"], dtype=np.float64)
                distance = float(np.linalg.norm(transformed_spine - target_center))
                if distance > maximum_distance_px:
                    continue
                dendrite_distance = 0.0
                if parent_id:
                    distance_map = distance_cache[(specimen_index, parent_id)]
                    x = int(round(float(transformed_parent[0])))
                    y = int(round(float(transformed_parent[1])))
                    if 0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1]:
                        dendrite_distance = float(distance_map[y, x])
                    else:
                        dendrite_distance = maximum_distance_px * 3.0
                if dendrite_distance > maximum_distance_px * 3.0:
                    continue
                geometric_distances[calcium_index, target_index] = distance
                dendrite_distances[calcium_index, target_index] = dendrite_distance
                costs[calcium_index, target_index] = distance + 0.20 * dendrite_distance
        row_ids, column_ids = linear_sum_assignment(costs)
        for calcium_index, target_index in zip(row_ids, column_ids):
            if costs[calcium_index, target_index] >= huge:
                continue
            calcium = calcium_rows[int(calcium_index)]
            registration, target = target_values[int(target_index)]
            row_costs = costs[int(calcium_index)]
            alternatives = np.delete(row_costs, int(target_index))
            alternatives = alternatives[alternatives < huge]
            margin = (
                float(np.min(alternatives) - row_costs[int(target_index)])
                if len(alternatives)
                else maximum_distance_px
            )
            distance = float(geometric_distances[int(calcium_index), int(target_index)])
            if distance <= maximum_distance_px * 0.35 and margin >= maximum_distance_px * 0.35:
                confidence = "high"
            elif distance <= maximum_distance_px * 0.70 and margin >= maximum_distance_px * 0.15:
                confidence = "medium"
            else:
                confidence = "low"
            proposed.append(
                {
                    "calcium_uid": str(calcium["calcium_uid"]),
                    "recording_id": recording_id,
                    "calcium_spine_roi": str(calcium["spine_roi"]),
                    "calcium_parent_segment": str(
                        calcium.get("parent_segment", "")
                    ),
                    "registration_id": str(registration["registration_id"]),
                    "synpo_specimen_index": int(registration["synpo_specimen_index"]),
                    "synpo_dendrite_id": int(target.get("dendrite_id") or 0),
                    "synpo_spine_id": int(target["spine_id"]),
                    "distance_px": distance,
                    "parent_dendrite_distance_px": float(
                        dendrite_distances[int(calcium_index), int(target_index)]
                    ),
                    "assignment_margin_cost": margin,
                    "automatic_confidence": confidence,
                    "review_state": "unreviewed",
                    "note": "",
                    "updated_at": _utc_timestamp(),
                }
            )
            used_targets.add(
                (int(registration["synpo_specimen_index"]), int(target["spine_id"]))
            )
    dataset["matches"] = preserved + proposed
    eligible_uids = {
        str(row.get("calcium_uid"))
        for row in _calcium_spines(dataset)
        if calcium_spine_in_review_scope(row)
    }
    matched_uids = {str(row.get("calcium_uid")) for row in dataset["matches"]}
    return {
        "proposed": len(proposed),
        "preserved_reviewed": len(preserved),
        "unmatched": len(eligible_uids - matched_uids),
        "outside_synpo_fields": sum(
            not calcium_spine_in_review_scope(row)
            for row in _calcium_spines(dataset)
        ),
    }


def assign_manual_match(
    dataset: dict[str, object],
    calcium_uid: str,
    synpo_specimen_index: int,
    synpo_spine_id: int,
    *,
    review_state: str = "confirmed",
    note: str = "",
    synpo_dendrite_id: int | None = None,
) -> dict[str, object]:
    if review_state not in REVIEW_STATES - {"ignored"}:
        raise ValueError("Manual match state must be confirmed, probable, ambiguous, or unreviewed.")
    calcium = _find_calcium_spine(dataset, calcium_uid)
    for row in dataset.setdefault("matches", []):
        if (
            str(row.get("calcium_uid")) != str(calcium_uid)
            and row.get("synpo_specimen_index") is not None
            and int(row["synpo_specimen_index"]) == int(synpo_specimen_index)
            and int(row["synpo_spine_id"]) == int(synpo_spine_id)
        ):
            raise ValueError("That Synpo spine is already assigned to another calcium ROI.")
    dataset["matches"] = [
        row for row in dataset["matches"] if str(row.get("calcium_uid")) != str(calcium_uid)
    ]
    match = {
        "calcium_uid": str(calcium_uid),
        "recording_id": str(calcium["recording_id"]),
        "calcium_spine_roi": str(calcium["spine_roi"]),
        "calcium_parent_segment": str(calcium.get("parent_segment", "")),
        "registration_id": None,
        "synpo_specimen_index": int(synpo_specimen_index),
        "synpo_dendrite_id": (
            int(synpo_dendrite_id) if synpo_dendrite_id is not None else None
        ),
        "synpo_spine_id": int(synpo_spine_id),
        "distance_px": None,
        "parent_dendrite_distance_px": None,
        "assignment_margin_cost": None,
        "automatic_confidence": "manual",
        "review_state": review_state,
        "note": str(note),
        "updated_at": _utc_timestamp(),
    }
    dataset["matches"].append(match)
    return match


def _clear_stale_match_state(match: dict[str, object]) -> None:
    for key in (
        "review_state_before_synpo_change",
        "synpo_result_stale",
        "synpo_result_stale_reason",
        "synpo_result_stale_at",
    ):
        match.pop(key, None)


def set_match_review(
    dataset: dict[str, object], calcium_uid: str, review_state: str, note: str = ""
) -> dict[str, object]:
    if review_state not in REVIEW_STATES:
        raise ValueError("Unknown calcium-match review state.")
    match = next(
        (row for row in dataset.setdefault("matches", []) if str(row.get("calcium_uid")) == str(calcium_uid)),
        None,
    )
    if match is None:
        if review_state != "ignored":
            raise ValueError("This calcium spine has no proposed match to review.")
        calcium = _find_calcium_spine(dataset, calcium_uid)
        match = {
            "calcium_uid": str(calcium_uid),
            "recording_id": str(calcium["recording_id"]),
            "calcium_spine_roi": str(calcium["spine_roi"]),
            "calcium_parent_segment": str(calcium.get("parent_segment", "")),
            "synpo_specimen_index": None,
            "synpo_dendrite_id": None,
            "synpo_spine_id": None,
            "automatic_confidence": "unmatched",
        }
        dataset["matches"].append(match)
    match["review_state"] = review_state
    match["note"] = str(note)
    match["updated_at"] = _utc_timestamp()
    if review_state != "unreviewed":
        _clear_stale_match_state(match)
    if review_state == "ignored":
        match["synpo_specimen_index"] = None
        match["synpo_dendrite_id"] = None
        match["synpo_spine_id"] = None
        match["registration_id"] = None
        match["distance_px"] = None
        match["parent_dendrite_distance_px"] = None
        match["assignment_margin_cost"] = None
    return match


def ignore_all_unmatched(dataset: dict[str, object]) -> int:
    decided = {str(row.get("calcium_uid")) for row in dataset.setdefault("matches", [])}
    count = 0
    for calcium in _calcium_spines(dataset):
        uid = str(calcium["calcium_uid"])
        if uid in decided or not calcium_spine_in_review_scope(calcium):
            continue
        set_match_review(dataset, uid, "ignored", "Ignored in bulk: no Synpo match")
        count += 1
    return count


def review_all_proposed(dataset: dict[str, object], state: str = "confirmed") -> int:
    if state not in {"confirmed", "probable", "ambiguous"}:
        raise ValueError("Bulk review state must be confirmed, probable, or ambiguous.")
    calcium_by_uid = {
        str(row.get("calcium_uid")): row for row in _calcium_spines(dataset)
    }
    count = 0
    for match in dataset.setdefault("matches", []):
        calcium = calcium_by_uid.get(str(match.get("calcium_uid")))
        if (
            str(match.get("review_state", "unreviewed")) == "unreviewed"
            and match.get("synpo_spine_id") is not None
            and calcium is not None
            and calcium_spine_in_review_scope(calcium)
        ):
            match["review_state"] = state
            match["updated_at"] = _utc_timestamp()
            _clear_stale_match_state(match)
            count += 1
    return count


def merge_calcium_features(
    manifest: dict[str, object],
    rows: list[dict[str, object]],
    dataset_id: str,
    accepted_states: Iterable[str] = ("confirmed",),
) -> list[dict[str, object]]:
    dataset = load_calcium_dataset(manifest, dataset_id)
    states = {str(value) for value in accepted_states}
    calcium_by_uid = {
        str(row["calcium_uid"]): row for row in _calcium_spines(dataset)
    }
    values: dict[tuple[str, str, int], dict[str, object]] = {}
    for match in dataset.get("matches", []):
        if (
            str(match.get("review_state")) not in states
            or match.get("synpo_spine_id") is None
            or bool(match.get("synpo_result_stale", False))
        ):
            continue
        specimen_index = int(match["synpo_specimen_index"])
        if not 0 <= specimen_index < len(manifest.get("specimens", [])):
            continue
        specimen = manifest["specimens"][specimen_index]
        calcium = calcium_by_uid.get(str(match["calcium_uid"]))
        if calcium is None or not calcium_spine_in_review_scope(calcium):
            continue
        values[
            (
                str(specimen.get("experimental_group", "")),
                str(specimen.get("specimen_id", "")),
                int(match["synpo_spine_id"]),
            )
        ] = {
            "calcium_dataset_id": dataset_id,
            "calcium_experiment": dataset.get("experiment_label"),
            "calcium_recording_id": calcium.get("recording_id"),
            "calcium_spine_roi": calcium.get("spine_roi"),
            "calcium_parent_segment": calcium.get("parent_segment"),
            "calcium_match_state": match.get("review_state"),
            "calcium_match_distance_px": match.get("distance_px"),
            "calcium_synpo_dendrite_id": match.get("synpo_dendrite_id"),
            "calcium_spine_peak_dff": calcium.get("calcium_spine_peak_dff"),
            "calcium_pearson_r": calcium.get("calcium_pearson_r"),
            **{
                f"calcium_metric_{key}": value
                for key, value in dict(calcium.get("metrics", {})).items()
            },
        }
    merged: list[dict[str, object]] = []
    for row in rows:
        key = (
            str(row.get("experimental_group", "")),
            str(row.get("specimen_id", "")),
            int(row.get("spine_id", 0)),
        )
        calcium_values = values.get(key, {})
        previous_dataset = str(row.get("calcium_dataset_id", ""))
        if calcium_values and previous_dataset and previous_dataset != dataset_id:
            raise ValueError(
                "The same Synpo spine is assigned in more than one selected calcium "
                "dataset. Review the calcium matches before combining experiments: "
                f"{key[0]} / {key[1]} / spine {key[2]}."
            )
        merged.append({**dict(row), **calcium_values})
    return merged


def calcium_analysis_rows(
    manifest: dict[str, object],
    dataset_id: str,
    morphology_run_id: str,
    accepted_states: Iterable[str] = ("confirmed",),
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    from .morphology import load_morphology_run

    run = load_morphology_run(manifest, morphology_run_id)
    assignments = merge_calcium_features(
        manifest,
        [dict(row) for row in run.get("assignments", [])],
        dataset_id,
        accepted_states,
    )
    matched = [
        row
        for row in assignments
        if str(row.get("calcium_dataset_id", "")) == str(dataset_id)
    ]
    matched_keys = {
        (
            str(row.get("experimental_group", "")),
            str(row.get("specimen_id", "")),
            int(row.get("spine_id", 0)),
        )
        for row in matched
    }
    dataset = load_calcium_dataset(manifest, dataset_id)
    audit: list[dict[str, object]] = []
    calcium_by_uid = {str(row["calcium_uid"]): row for row in _calcium_spines(dataset)}
    match_by_uid = {
        str(row.get("calcium_uid")): row for row in dataset.get("matches", [])
    }
    states = {str(value) for value in accepted_states}
    for calcium_uid, calcium in calcium_by_uid.items():
        match = match_by_uid.get(calcium_uid)
        match_values = dict(match or {})
        specimen_index = match_values.get("synpo_specimen_index")
        specimen = (
            manifest["specimens"][int(specimen_index)]
            if specimen_index is not None and 0 <= int(specimen_index) < len(manifest["specimens"])
            else {}
        )
        key = (
            str(specimen.get("experimental_group", "")),
            str(specimen.get("specimen_id", "")),
            int(match_values.get("synpo_spine_id") or 0),
        )
        state = str(match_values.get("review_state", "unmatched"))
        reason = "included"
        if not calcium_spine_in_review_scope(calcium):
            reason = "outside every registered Synpo field (automatically excluded)"
        elif match is None:
            reason = "no saved match proposal or decision"
        elif state not in states:
            reason = f"match state {state!r} not selected"
        elif match_values.get("synpo_spine_id") is None:
            reason = "no Synpo spine assigned"
        elif key not in matched_keys:
            reason = "Synpo spine is not included in the selected morphology run"
        audit.append(
            {
                **match_values,
                "calcium_uid": calcium_uid,
                "recording_id": calcium.get("recording_id"),
                "calcium_spine_roi": calcium.get("spine_roi"),
                "calcium_parent_segment": calcium.get("parent_segment"),
                "calcium_source_included": calcium.get("source_included"),
                "synpo_field_coverage": calcium.get("synpo_field_coverage"),
                "synpo_registered_field_count": calcium.get(
                    "synpo_registered_field_count"
                ),
                "calcium_spine_peak_dff": calcium.get("calcium_spine_peak_dff"),
                "calcium_pearson_r": calcium.get("calcium_pearson_r"),
                "experimental_group": specimen.get("experimental_group"),
                "specimen_id": specimen.get("specimen_id"),
                "analysis_status": reason,
            }
        )
    return matched, audit, run


def _bootstrap_by_specimen(
    rows: list[dict[str, object]], x_key: str, y_key: str, repetitions: int = 1000
) -> tuple[float | None, float | None]:
    specimens: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        specimens.setdefault(
            (str(row.get("experimental_group", "")), str(row.get("specimen_id", ""))), []
        ).append(row)
    if len(specimens) < 2:
        return None, None
    groups = list(specimens.values())
    generator = np.random.default_rng(42)
    correlations: list[float] = []
    for _ in range(repetitions):
        sampled = [groups[int(index)] for index in generator.integers(0, len(groups), len(groups))]
        values = [item for specimen_rows in sampled for item in specimen_rows]
        x = np.asarray([float(row[x_key]) for row in values], dtype=np.float64)
        y = np.asarray([float(row[y_key]) for row in values], dtype=np.float64)
        if len(x) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    if not correlations:
        return None, None
    return float(np.percentile(correlations, 2.5)), float(np.percentile(correlations, 97.5))


def calcium_cluster_statistics(
    rows: list[dict[str, object]],
    *,
    x_key: str = "protein_puncta_total_volume_um3",
    y_key: str = "calcium_spine_peak_dff",
) -> list[dict[str, object]]:
    clusters = sorted({int(row["morphology_cluster_id"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for cluster in clusters:
        members = [
            row
            for row in rows
            if int(row["morphology_cluster_id"]) == cluster
            and row.get(x_key) is not None
            and row.get(y_key) is not None
        ]
        x = np.asarray([float(row[x_key]) for row in members], dtype=np.float64)
        y = np.asarray([float(row[y_key]) for row in members], dtype=np.float64)
        pearson_r = pearson_p = spearman_r = spearman_p = slope = intercept = None
        if len(members) >= 2 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            pearson = stats.pearsonr(x, y)
            spearman = stats.spearmanr(x, y)
            regression = stats.linregress(x, y)
            pearson_r, pearson_p = float(pearson.statistic), float(pearson.pvalue)
            spearman_r, spearman_p = float(spearman.statistic), float(spearman.pvalue)
            slope, intercept = float(regression.slope), float(regression.intercept)
        ci_low, ci_high = _bootstrap_by_specimen(members, x_key, y_key)
        summaries.append(
            {
                "morphology_cluster_id": cluster,
                "spine_count": len(members),
                "specimen_count": len(
                    {
                        (str(row.get("experimental_group", "")), str(row.get("specimen_id", "")))
                        for row in members
                    }
                ),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_r,
                "spearman_p": spearman_p,
                "regression_slope": slope,
                "regression_intercept": intercept,
                "specimen_bootstrap_pearson_ci_low": ci_low,
                "specimen_bootstrap_pearson_ci_high": ci_high,
                "x_metric": x_key,
                "y_metric": y_key,
            }
        )
    return summaries


def draw_calcium_dependency(
    figure: Figure,
    rows: list[dict[str, object]],
    morphology_run: dict[str, object],
    *,
    x_key: str = "protein_puncta_total_volume_um3",
    y_key: str = "calcium_spine_peak_dff",
    included_groups: Iterable[str] = (),
    point_alpha: float = 0.72,
    log_x: bool = False,
    log_y: bool = False,
    show_regression: bool = True,
) -> list[dict[str, object]]:
    figure.clear()
    axis = figure.add_subplot(111)
    selected_groups = {str(value) for value in included_groups}
    filtered = [
        row
        for row in rows
        if (not selected_groups or str(row.get("experimental_group", "")) in selected_groups)
        and row.get(x_key) is not None
        and row.get(y_key) is not None
    ]
    colors = {
        int(row["morphology_cluster_id"]): str(row.get("color", "#457b9d"))
        for row in morphology_run.get("cluster_definitions", [])
    }
    groups = sorted({str(row.get("experimental_group", "")) for row in filtered})
    markers = ("o", "^", "s", "D", "P", "X", "v", "<", ">", "*")
    marker_by_group = {group: markers[index % len(markers)] for index, group in enumerate(groups)}
    statistics = calcium_cluster_statistics(filtered, x_key=x_key, y_key=y_key)
    for cluster in sorted({int(row["morphology_cluster_id"]) for row in filtered}):
        cluster_rows = [row for row in filtered if int(row["morphology_cluster_id"]) == cluster]
        for group in groups:
            members = [row for row in cluster_rows if str(row.get("experimental_group", "")) == group]
            if not members:
                continue
            axis.scatter(
                [float(row[x_key]) for row in members],
                [float(row[y_key]) for row in members],
                color=colors.get(cluster, "#457b9d"),
                marker=marker_by_group[group],
                alpha=max(0.0, min(1.0, float(point_alpha))),
                label=f"Cluster {cluster} · {group}",
            )
        summary = next(row for row in statistics if int(row["morphology_cluster_id"]) == cluster)
        if show_regression and summary.get("regression_slope") is not None:
            x = np.asarray([float(row[x_key]) for row in cluster_rows], dtype=np.float64)
            line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            line_y = float(summary["regression_intercept"]) + float(summary["regression_slope"]) * line_x
            axis.plot(line_x, line_y, color=colors.get(cluster, "#457b9d"), linewidth=1.6)
    if log_x:
        axis.set_xscale("symlog", linthresh=1e-9)
    if log_y:
        axis.set_yscale("symlog", linthresh=1e-9)
    labels = {
        "protein_puncta_total_volume_um3": "Protein puncta total volume (µm³)",
        "protein_puncta_count": "Protein puncta count",
        "protein_puncta_to_spine_volume_ratio": "Protein puncta / spine volume",
        "calcium_spine_peak_dff": "Spine peak ΔF/F0",
        "calcium_pearson_r": "Spine-parent Pearson r",
    }
    axis.set(
        xlabel=labels.get(x_key, x_key),
        ylabel=labels.get(y_key, y_key),
        title=f"{labels.get(y_key, y_key)} versus {labels.get(x_key, x_key)} by morphology cluster",
    )
    if filtered:
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    else:
        axis.text(0.5, 0.5, "No reviewed matched spines satisfy these filters.", ha="center", va="center", transform=axis.transAxes)
    axis.grid(alpha=0.22)
    figure.tight_layout()
    return statistics


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )


def _append_sheet(workbook: Workbook, title: str, rows: list[dict[str, object]]) -> None:
    sheet = workbook.create_sheet(title[:31])
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        sheet.append(["No rows"])
        return
    sheet.append(fields)
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, (list, tuple, dict)):
                value = json.dumps(value, ensure_ascii=False)
            values.append(value)
        sheet.append(values)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def export_calcium_analysis(
    manifest: dict[str, object],
    dataset_id: str,
    morphology_run_id: str,
    output_path: str | Path,
    *,
    accepted_states: Iterable[str] = ("confirmed",),
    x_key: str = "protein_puncta_total_volume_um3",
    y_key: str = "calcium_spine_peak_dff",
    included_groups: Iterable[str] = (),
    point_alpha: float = 0.72,
    log_x: bool = False,
    log_y: bool = False,
    show_regression: bool = True,
    analysis_name: str = "",
    analysis_run_id: str = "",
) -> Path:
    rows, audit, run = calcium_analysis_rows(
        manifest, dataset_id, morphology_run_id, accepted_states
    )
    dataset = load_calcium_dataset(manifest, dataset_id)
    calcium_rois = [
        {
            "calcium_uid": spine.get("calcium_uid"),
            "recording_id": spine.get("recording_id"),
            "spine_roi": spine.get("spine_roi"),
            "parent_segment": spine.get("parent_segment"),
            "biological_dendrite_id": spine.get("biological_dendrite_id"),
            "source_included": spine.get("source_included"),
            "synpo_field_coverage": spine.get("synpo_field_coverage"),
            "synpo_inside_registration_ids": spine.get(
                "synpo_inside_registration_ids"
            ),
            "synpo_registered_field_count": spine.get(
                "synpo_registered_field_count"
            ),
            "center_xy": spine.get("center_xy"),
            "parent_center_xy": spine.get("parent_center_xy"),
            "roi_points_xy": dict(spine.get("roi", {})).get("points_xy"),
            "calcium_spine_peak_dff": spine.get("calcium_spine_peak_dff"),
            "calcium_pearson_r": spine.get("calcium_pearson_r"),
            **{
                f"calcium_metric_{key}": value
                for key, value in dict(spine.get("metrics", {})).items()
            },
        }
        for recording in dataset.get("recordings", [])
        for spine in recording.get("spines", [])
    ]
    registrations = [dict(row) for row in dataset.get("registrations", [])]
    groups = tuple(str(value) for value in included_groups)
    selected_groups = set(groups)
    plotted_rows = [
        row
        for row in rows
        if (
            not selected_groups
            or str(row.get("experimental_group", "")) in selected_groups
        )
        and row.get(x_key) is not None
        and row.get(y_key) is not None
    ]
    plot_audit = []
    for row in rows:
        reason = "plotted"
        if selected_groups and str(row.get("experimental_group", "")) not in selected_groups:
            reason = "experimental group not selected"
        elif row.get(x_key) is None:
            reason = f"missing X metric: {x_key}"
        elif row.get(y_key) is None:
            reason = f"missing Y metric: {y_key}"
        plot_audit.append(
            {
                "experimental_group": row.get("experimental_group"),
                "specimen_id": row.get("specimen_id"),
                "spine_id": row.get("spine_id"),
                "calcium_uid": row.get("calcium_uid"),
                "morphology_cluster_id": row.get("morphology_cluster_id"),
                "plot_status": reason,
                "x_metric": x_key,
                "x_value": row.get(x_key),
                "y_metric": y_key,
                "y_value": row.get(y_key),
            }
        )
    statistics = calcium_cluster_statistics(
        plotted_rows, x_key=x_key, y_key=y_key
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.casefold() != ".xlsx":
        destination = destination.with_suffix(".xlsx")
    destination.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.remove(workbook.active)
    _append_sheet(workbook, "Matched_Spines", rows)
    _append_sheet(workbook, "Plotted_Subset", plotted_rows)
    _append_sheet(workbook, "Cluster_Statistics", statistics)
    _append_sheet(workbook, "Match_Audit", audit)
    _append_sheet(workbook, "Plot_Audit", plot_audit)
    _append_sheet(workbook, "Calcium_ROIs", calcium_rois)
    _append_sheet(workbook, "Registrations", registrations)
    settings = [
        {"setting": "analysis_name", "value": analysis_name},
        {"setting": "analysis_run_id", "value": analysis_run_id},
        {"setting": "dataset_id", "value": dataset_id},
        {"setting": "morphology_run_id", "value": morphology_run_id},
        {"setting": "accepted_states", "value": ", ".join(accepted_states)},
        {"setting": "x_metric", "value": x_key},
        {"setting": "y_metric", "value": y_key},
        {"setting": "included_groups", "value": ", ".join(groups)},
        {"setting": "point_alpha", "value": point_alpha},
        {"setting": "log_x", "value": log_x},
        {"setting": "log_y", "value": log_y},
        {"setting": "show_regression", "value": show_regression},
    ]
    _append_sheet(workbook, "Analysis_Settings", settings)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    workbook.save(temporary)
    os.replace(temporary, destination)
    csv_root = destination.with_name(destination.stem + "_csv")
    _write_csv(csv_root / "matched_spines.csv", rows)
    _write_csv(csv_root / "plotted_subset.csv", plotted_rows)
    _write_csv(csv_root / "cluster_statistics.csv", statistics)
    _write_csv(csv_root / "match_audit.csv", audit)
    _write_csv(csv_root / "plot_audit.csv", plot_audit)
    _write_csv(csv_root / "calcium_rois.csv", calcium_rois)
    _write_csv(csv_root / "registrations.csv", registrations)
    figure = Figure(figsize=(9.5, 6.5), tight_layout=True)
    draw_calcium_dependency(
        figure,
        rows,
        run,
        x_key=x_key,
        y_key=y_key,
        included_groups=groups,
        point_alpha=point_alpha,
        log_x=log_x,
        log_y=log_y,
        show_regression=show_regression,
    )
    figure.savefig(destination.with_name(destination.stem + "_dependency.svg"), bbox_inches="tight")
    figure.savefig(destination.with_name(destination.stem + "_dependency.pdf"), bbox_inches="tight")
    figure.savefig(destination.with_name(destination.stem + "_dependency.png"), dpi=600, bbox_inches="tight")
    return destination
