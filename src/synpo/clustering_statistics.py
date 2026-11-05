from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Callable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ClusteringStatisticsSettings:
    """Settings for comparisons performed after morphology clustering."""

    analysis_level: str = "specimen"
    included_groups: tuple[str, ...] = ()
    confidence_level: float = 0.95
    permutation_repetitions: int = 10_000
    random_seed: int = 42
    continuous_test: str = "welch"
    metrics: tuple[str, ...] = ()
    pie_labels: str = "percent"

    def validate(self) -> None:
        if self.analysis_level not in {"spine", "specimen", "neuron"}:
            raise ValueError("Analysis level must be spine, specimen, or neuron.")
        if not 0.90 <= self.confidence_level <= 0.99:
            raise ValueError("Confidence level must be between 90% and 99%.")
        if not 100 <= self.permutation_repetitions <= 100_000:
            raise ValueError("Permutation repetitions must be between 100 and 100,000.")
        if self.continuous_test not in {"welch", "mann_whitney", "permutation"}:
            raise ValueError("Unknown continuous-metric test.")
        if self.pie_labels not in {"percent", "count", "both", "none"}:
            raise ValueError("Unknown pie-label mode.")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object] | None) -> "ClusteringStatisticsSettings":
        source = value or {}
        settings = cls(
            analysis_level=str(source.get("analysis_level", "specimen")),
            included_groups=tuple(str(item) for item in source.get("included_groups", ())),
            confidence_level=float(source.get("confidence_level", 0.95)),
            permutation_repetitions=int(source.get("permutation_repetitions", 10_000)),
            random_seed=int(source.get("random_seed", 42)),
            continuous_test=str(source.get("continuous_test", "welch")),
            metrics=tuple(str(item) for item in source.get("metrics", ())),
            pie_labels=str(source.get("pie_labels", "percent")),
        )
        settings.validate()
        return settings


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _metric_value(row: dict[str, object], metric: str) -> float | None:
    stored = _finite(row.get(f"clustering_feature_{metric}"))
    if stored is not None:
        return stored
    columns = {
        "log_volume": "volume_um3",
        "linear_volume": "volume_um3",
        "sqrt_volume": "volume_um3",
        "curvilinear_length": "spine_curvilinear_length_um",
        "base_to_tip_distance": "spine_base_to_tip_distance_um",
        "tortuosity": "centerline_tortuosity",
        "maximum_width": "maximum_width_um",
        "elongation": "principal_axis_elongation",
        "head_width": "head_maximum_width_um",
        "neck_width": "neck_median_width_um",
        "head_neck_ratio": "head_to_neck_width_ratio",
        "head_volume": "head_volume_um3",
        "neck_volume": "neck_volume_um3",
        "surface_area": "surface_area_um2",
        "sphericity": "sphericity",
        "protein_puncta_count": "protein_puncta_count",
        "protein_puncta_total_volume": "protein_puncta_total_volume_um3",
        "protein_puncta_volume_ratio": "protein_puncta_to_spine_volume_ratio",
        "protein_puncta_mean_volume": "protein_puncta_mean_volume_um3",
        "protein_puncta_axis_position": "protein_puncta_axis_centroid_fraction",
        "calcium_peak_dff": "calcium_spine_peak_dff",
        "calcium_pearson_r": "calcium_pearson_r",
    }
    value = _finite(row.get(columns.get(metric, metric)))
    if value is None:
        return None
    if metric == "log_volume":
        return math.log10(value) if value > 0 else None
    if metric == "sqrt_volume":
        return math.sqrt(value) if value >= 0 else None
    return value


def _describe(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)) if len(array) else None,
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else None,
        "sem": float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else None,
        "median": float(np.median(array)) if len(array) else None,
        "q1": float(np.percentile(array, 25)) if len(array) else None,
        "q3": float(np.percentile(array, 75)) if len(array) else None,
    }


def _bh(rows: list[dict[str, object]], key: str = "p_value") -> None:
    indexed = [
        (index, float(row[key]))
        for index, row in enumerate(rows)
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    indexed.sort(key=lambda item: item[1])
    total = len(indexed)
    running = 1.0
    adjusted: dict[int, float] = {}
    for rank, (index, p_value) in reversed(list(enumerate(indexed, start=1))):
        running = min(running, p_value * total / rank)
        adjusted[index] = min(1.0, running)
    for index, row in enumerate(rows):
        row["bh_q_value"] = adjusted.get(index)


def _agresti_caffo(
    x_first: int,
    n_first: int,
    x_second: int,
    n_second: int,
    confidence: float,
) -> dict[str, float | None]:
    if n_first <= 0 or n_second <= 0:
        return {
            "observed_difference": None,
            "adjusted_difference": None,
            "ci_low": None,
            "ci_high": None,
            "z_score": None,
            "p_value": None,
        }
    observed = x_first / n_first - x_second / n_second
    first = (x_first + 1.0) / (n_first + 2.0)
    second = (x_second + 1.0) / (n_second + 2.0)
    difference = first - second
    standard_error = math.sqrt(
        first * (1.0 - first) / (n_first + 2.0)
        + second * (1.0 - second) / (n_second + 2.0)
    )
    critical = float(stats.norm.ppf(0.5 + confidence / 2.0))
    z_score = difference / standard_error if standard_error > 0 else 0.0
    return {
        "observed_difference": float(observed),
        "adjusted_difference": float(difference),
        "ci_low": float(difference - critical * standard_error),
        "ci_high": float(difference + critical * standard_error),
        "z_score": float(z_score),
        "p_value": float(2.0 * stats.norm.sf(abs(z_score))),
    }


def _unit_id(row: dict[str, object], level: str) -> str:
    if level == "spine":
        return "|".join(
            str(row.get(key, ""))
            for key in ("experimental_group", "specimen_id", "spine_id")
        )
    if level == "neuron":
        return str(row.get("neuron_id") or row.get("specimen_id") or "")
    return str(row.get("specimen_id") or "")


def _permutation_difference(
    first: np.ndarray,
    second: np.ndarray,
    repetitions: int,
    random: np.random.Generator,
) -> tuple[float, float]:
    observed = float(np.mean(first) - np.mean(second))
    joined = np.concatenate((first, second))
    first_size = len(first)
    exceed = 0
    completed = 0
    total = float(np.sum(joined))
    # Batch random label assignments. This keeps neuron/specimen permutations fast
    # without allocating a repetitions-by-spines matrix for unusually large inputs.
    batch_limit = max(1, 2_000_000 // max(1, len(joined)))
    while completed < repetitions:
        batch = min(batch_limit, repetitions - completed)
        priorities = random.random((batch, len(joined)))
        selected = np.argpartition(
            priorities, first_size - 1, axis=1
        )[:, :first_size]
        first_sums = np.sum(joined[selected], axis=1)
        candidates = (
            first_sums / first_size
            - (total - first_sums) / len(second)
        )
        exceed += int(
            np.count_nonzero(np.abs(candidates) >= abs(observed) - 1e-15)
        )
        completed += batch
    return observed, float((exceed + 1) / (repetitions + 1))


def _metric_test(
    first: np.ndarray,
    second: np.ndarray,
    method: str,
    repetitions: int,
    random: np.random.Generator,
) -> tuple[str, float, float]:
    difference = float(np.mean(first) - np.mean(second))
    if method == "mann_whitney":
        statistic, p_value = stats.mannwhitneyu(first, second, alternative="two-sided")
        return "Mann-Whitney U", float(statistic), float(p_value)
    if method == "permutation":
        observed, p_value = _permutation_difference(
            first, second, repetitions, random
        )
        return "Monte Carlo permutation of mean difference", observed, p_value
    statistic, p_value = stats.ttest_ind(first, second, equal_var=False)
    return "Welch independent-samples t-test", float(statistic), float(p_value)


def calculate_clustering_statistics(
    result: dict[str, object],
    settings: ClusteringStatisticsSettings,
    *,
    metric_labels: dict[str, str] | None = None,
) -> dict[str, object]:
    """Calculate composition and metric comparisons for one saved clustering run.

    The pooled chi-square and Agresti-Caffo rows deliberately retain their
    spine-independence assumption. At specimen or neuron level, additional
    permutation rows are calculated after equal-weight aggregation.
    """

    settings.validate()
    assignments = [dict(row) for row in result.get("assignments", [])]
    available_groups = sorted(
        {str(row.get("experimental_group", "")) for row in assignments}
    )
    selected_groups = (
        [group for group in settings.included_groups if group in available_groups]
        if settings.included_groups
        else available_groups
    )
    rows = [
        row
        for row in assignments
        if str(row.get("experimental_group", "")) in selected_groups
    ]
    if not rows:
        raise ValueError("No spines belong to the selected statistical groups.")
    clusters = sorted({int(row["morphology_cluster_id"]) for row in rows})
    warnings: list[str] = []
    if settings.analysis_level == "spine":
        warnings.append(
            "Spine-level inferential tests treat spines as independent and may be "
            "pseudoreplicated when several spines come from one specimen or neuron."
        )
    if len(selected_groups) < 2:
        warnings.append(
            "Only one experimental group is selected; descriptive summaries are "
            "shown and between-group tests are suppressed."
        )

    totals = {
        group: sum(str(row.get("experimental_group", "")) == group for row in rows)
        for group in selected_groups
    }
    composition: list[dict[str, object]] = []
    contingency = np.zeros((len(selected_groups), len(clusters)), dtype=np.int64)
    for group_index, group in enumerate(selected_groups):
        for cluster_index, cluster in enumerate(clusters):
            count = sum(
                str(row.get("experimental_group", "")) == group
                and int(row["morphology_cluster_id"]) == cluster
                for row in rows
            )
            contingency[group_index, cluster_index] = count
            composition.append(
                {
                    "experimental_group": group,
                    "morphology_cluster_id": cluster,
                    "spine_count": int(count),
                    "group_spine_count": int(totals[group]),
                    "percent_within_group": (
                        100.0 * count / totals[group] if totals[group] else None
                    ),
                }
            )

    omnibus: list[dict[str, object]] = []
    if len(selected_groups) >= 2 and len(clusters) >= 2:
        chi2, p_value, degrees, expected = stats.chi2_contingency(
            contingency, correction=False
        )
        minimum_expected = float(np.min(expected))
        sparse_count = int(np.count_nonzero(expected < 5.0))
        if sparse_count:
            warnings.append(
                f"Pearson chi-square has {sparse_count} expected cell(s) below 5 "
                f"(minimum {minimum_expected:.3g}); interpret its asymptotic p-value cautiously."
            )
        denominator = int(np.sum(contingency)) * min(
            len(selected_groups) - 1, len(clusters) - 1
        )
        omnibus.append(
            {
                "test": "Pearson chi-square group x cluster",
                "analysis_level": "spine (pooled; independence assumed)",
                "chi_square": float(chi2),
                "degrees_of_freedom": int(degrees),
                "p_value": float(p_value),
                "cramers_v": float(math.sqrt(chi2 / denominator)) if denominator else None,
                "minimum_expected_count": minimum_expected,
                "expected_cells_below_5": sparse_count,
                "warning": "Sparse expected counts" if sparse_count else "",
            }
        )

    agresti: list[dict[str, object]] = []
    for first_group, second_group in combinations(selected_groups, 2):
        for cluster in clusters:
            x_first = int(
                contingency[selected_groups.index(first_group), clusters.index(cluster)]
            )
            x_second = int(
                contingency[selected_groups.index(second_group), clusters.index(cluster)]
            )
            row = {
                "first_group": first_group,
                "second_group": second_group,
                "direction": "first_group minus second_group",
                "morphology_cluster_id": cluster,
                "first_cluster_count": x_first,
                "first_group_total": totals[first_group],
                "second_cluster_count": x_second,
                "second_group_total": totals[second_group],
                "confidence_level": settings.confidence_level,
                "analysis_level": "spine (pooled; independence assumed)",
                **_agresti_caffo(
                    x_first,
                    totals[first_group],
                    x_second,
                    totals[second_group],
                    settings.confidence_level,
                ),
            }
            agresti.append(row)
    _bh(agresti)

    unit_composition: list[dict[str, object]] = []
    unit_permutation: list[dict[str, object]] = []
    if settings.analysis_level in {"specimen", "neuron"}:
        unit_rows: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in rows:
            group = str(row.get("experimental_group", ""))
            unit_rows.setdefault((group, _unit_id(row, settings.analysis_level)), []).append(row)
        for (group, unit), members in sorted(unit_rows.items()):
            for cluster in clusters:
                fraction = sum(
                    int(row["morphology_cluster_id"]) == cluster for row in members
                ) / len(members)
                unit_composition.append(
                    {
                        "analysis_level": settings.analysis_level,
                        "experimental_group": group,
                        "unit_id": unit,
                        "morphology_cluster_id": cluster,
                        "unit_spine_count": len(members),
                        "cluster_fraction": float(fraction),
                        "cluster_percent": float(100.0 * fraction),
                    }
                )
        random = np.random.default_rng(settings.random_seed)
        for first_group, second_group in combinations(selected_groups, 2):
            for cluster in clusters:
                first = np.asarray(
                    [
                        row["cluster_fraction"]
                        for row in unit_composition
                        if row["experimental_group"] == first_group
                        and row["morphology_cluster_id"] == cluster
                    ],
                    dtype=np.float64,
                )
                second = np.asarray(
                    [
                        row["cluster_fraction"]
                        for row in unit_composition
                        if row["experimental_group"] == second_group
                        and row["morphology_cluster_id"] == cluster
                    ],
                    dtype=np.float64,
                )
                if len(first) < 2 or len(second) < 2:
                    continue
                effect, p_value = _permutation_difference(
                    first,
                    second,
                    settings.permutation_repetitions,
                    random,
                )
                unit_permutation.append(
                    {
                        "analysis_level": settings.analysis_level,
                        "first_group": first_group,
                        "second_group": second_group,
                        "direction": "first_group minus second_group",
                        "morphology_cluster_id": cluster,
                        "first_unit_count": len(first),
                        "second_unit_count": len(second),
                        "mean_fraction_difference": effect,
                        "p_value": p_value,
                        "permutation_repetitions": settings.permutation_repetitions,
                        "random_seed": settings.random_seed,
                    }
                )
        _bh(unit_permutation)

    labels = metric_labels or {}
    selected_metrics = list(dict.fromkeys(settings.metrics))
    clustering_inputs = set(result.get("settings", {}).get("features", []))
    metric_descriptives: list[dict[str, object]] = []
    metric_comparisons: list[dict[str, object]] = []
    random = np.random.default_rng(settings.random_seed + 193)
    for cluster in clusters:
        cluster_rows = [
            row for row in rows if int(row["morphology_cluster_id"]) == cluster
        ]
        for metric in selected_metrics:
            values_by_group: dict[str, np.ndarray] = {}
            for group in selected_groups:
                group_rows = [
                    row
                    for row in cluster_rows
                    if str(row.get("experimental_group", "")) == group
                ]
                spine_values = [
                    value
                    for value in (_metric_value(row, metric) for row in group_rows)
                    if value is not None
                ]
                if settings.analysis_level == "spine":
                    analysis_values = spine_values
                    unit_count = len(spine_values)
                else:
                    by_unit: dict[str, list[float]] = {}
                    for row in group_rows:
                        value = _metric_value(row, metric)
                        if value is not None:
                            by_unit.setdefault(
                                _unit_id(row, settings.analysis_level), []
                            ).append(value)
                    analysis_values = [
                        float(np.mean(unit_values))
                        for unit_values in by_unit.values()
                    ]
                    unit_count = len(by_unit)
                values_by_group[group] = np.asarray(
                    analysis_values, dtype=np.float64
                )
                metric_descriptives.append(
                    {
                        "analysis_level": settings.analysis_level,
                        "independence_assumption": (
                            "spines treated as independent"
                            if settings.analysis_level == "spine"
                            else f"equal-weight {settings.analysis_level} aggregates"
                        ),
                        "experimental_group": group,
                        "morphology_cluster_id": cluster,
                        "metric": metric,
                        "metric_label": labels.get(metric, metric),
                        "spine_value_count": len(spine_values),
                        "analysis_unit_count": unit_count,
                        "inferential_status": (
                            "descriptive only: metric was a clustering input"
                            if metric in clustering_inputs
                            else "eligible for between-group inference"
                        ),
                        **_describe(analysis_values),
                    }
                )
            if metric in clustering_inputs:
                continue
            for first_group, second_group in combinations(selected_groups, 2):
                first = values_by_group[first_group]
                second = values_by_group[second_group]
                if len(first) < 2 or len(second) < 2:
                    continue
                method = (
                    "permutation"
                    if settings.analysis_level == "neuron"
                    else settings.continuous_test
                )
                test_name, statistic, p_value = _metric_test(
                    first,
                    second,
                    method,
                    settings.permutation_repetitions,
                    random,
                )
                metric_comparisons.append(
                    {
                        "analysis_level": settings.analysis_level,
                        "independence_assumption": (
                            "spines treated as independent"
                            if settings.analysis_level == "spine"
                            else f"equal-weight {settings.analysis_level} aggregates"
                        ),
                        "morphology_cluster_id": cluster,
                        "metric": metric,
                        "metric_label": labels.get(metric, metric),
                        "first_group": first_group,
                        "second_group": second_group,
                        "direction": "first_group minus second_group",
                        "first_unit_count": len(first),
                        "second_unit_count": len(second),
                        "first_mean": float(np.mean(first)),
                        "second_mean": float(np.mean(second)),
                        "mean_difference": float(np.mean(first) - np.mean(second)),
                        "test": test_name,
                        "statistic": statistic,
                        "p_value": p_value,
                    }
                )
    _bh(metric_comparisons)

    protein_presence: list[dict[str, object]] = []
    protein_presence_unit: list[dict[str, object]] = []
    for cluster in clusters:
        members = [row for row in rows if int(row["morphology_cluster_id"]) == cluster]
        for first_group, second_group in combinations(selected_groups, 2):
            first_rows = [row for row in members if str(row.get("experimental_group", "")) == first_group]
            second_rows = [row for row in members if str(row.get("experimental_group", "")) == second_group]
            row = {
                "morphology_cluster_id": cluster,
                "first_group": first_group,
                "second_group": second_group,
                "direction": "first_group minus second_group",
                "first_positive": sum(bool(item.get("has_protein_cluster", False)) for item in first_rows),
                "first_total": len(first_rows),
                "second_positive": sum(bool(item.get("has_protein_cluster", False)) for item in second_rows),
                "second_total": len(second_rows),
                "analysis_level": "spine (pooled; independence assumed)",
                **_agresti_caffo(
                    sum(bool(item.get("has_protein_cluster", False)) for item in first_rows),
                    len(first_rows),
                    sum(bool(item.get("has_protein_cluster", False)) for item in second_rows),
                    len(second_rows),
                    settings.confidence_level,
                ),
            }
            protein_presence.append(row)
    _bh(protein_presence)
    if settings.analysis_level in {"specimen", "neuron"}:
        random = np.random.default_rng(settings.random_seed + 389)
        for cluster in clusters:
            cluster_rows = [
                row for row in rows if int(row["morphology_cluster_id"]) == cluster
            ]
            fractions_by_group: dict[str, np.ndarray] = {}
            for group in selected_groups:
                by_unit: dict[str, list[bool]] = {}
                for row in cluster_rows:
                    if str(row.get("experimental_group", "")) != group:
                        continue
                    by_unit.setdefault(
                        _unit_id(row, settings.analysis_level), []
                    ).append(bool(row.get("has_protein_cluster", False)))
                fractions_by_group[group] = np.asarray(
                    [float(np.mean(values)) for values in by_unit.values()],
                    dtype=np.float64,
                )
            for first_group, second_group in combinations(selected_groups, 2):
                first = fractions_by_group[first_group]
                second = fractions_by_group[second_group]
                if len(first) < 2 or len(second) < 2:
                    continue
                difference, p_value = _permutation_difference(
                    first,
                    second,
                    settings.permutation_repetitions,
                    random,
                )
                protein_presence_unit.append(
                    {
                        "analysis_level": settings.analysis_level,
                        "morphology_cluster_id": cluster,
                        "first_group": first_group,
                        "second_group": second_group,
                        "direction": "first_group minus second_group",
                        "first_unit_count": len(first),
                        "second_unit_count": len(second),
                        "mean_positive_fraction_difference": difference,
                        "test": "Monte Carlo permutation of mean unit fraction",
                        "p_value": p_value,
                        "permutation_repetitions": settings.permutation_repetitions,
                        "random_seed": settings.random_seed,
                    }
                )
        _bh(protein_presence_unit)

    return {
        "created_at": time.time(),
        "settings": settings.to_dict(),
        "groups": selected_groups,
        "clusters": clusters,
        "warnings": warnings,
        "composition_summary": composition,
        "composition_omnibus": omnibus,
        "agresti_caffo": agresti,
        "unit_composition": unit_composition,
        "unit_permutation": unit_permutation,
        "metric_descriptives": metric_descriptives,
        "metric_comparisons": metric_comparisons,
        "protein_presence_comparisons": protein_presence,
        "protein_presence_unit_permutation": protein_presence_unit,
        "multiple_testing_families": {
            "cluster_composition_agresti_caffo": len(agresti),
            "unit_cluster_fraction_permutations": len(unit_permutation),
            "continuous_metric_comparisons": len(metric_comparisons),
            "protein_presence_agresti_caffo": len(protein_presence),
            "protein_presence_unit_permutations": len(protein_presence_unit),
        },
    }


def _pie_autopct(mode: str, total: int) -> Callable[[float], str] | None:
    if mode == "none":
        return None

    def label(percent: float) -> str:
        count = int(round(percent * total / 100.0))
        if mode == "count":
            return str(count)
        if mode == "both":
            return f"{percent:.1f}%\n(n={count})"
        return f"{percent:.1f}%"

    return label


def _comparison_lines(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return [
            "No inferential comparison was available.",
            "The metric may be a clustering input,",
            "or a group may have fewer than two units.",
        ]
    lines: list[str] = []
    for row in rows:
        effect = next(
            (
                row.get(key)
                for key in (
                    "mean_difference",
                    "mean_fraction_difference",
                    "mean_positive_fraction_difference",
                    "adjusted_difference",
                )
                if row.get(key) is not None
            ),
            None,
        )
        effect_text = f"; effect={float(effect):.4g}" if effect is not None else ""
        q_value = row.get("bh_q_value")
        q_text = f"; q={float(q_value):.3g}" if q_value is not None else ""
        p_value = row.get("p_value")
        p_text = f"{float(p_value):.3g}" if p_value is not None else "not available"
        lines.append(
            f"{row.get('first_group')} - {row.get('second_group')}: "
            f"p={p_text}{q_text}{effect_text}"
        )
    return lines


def _metric_analysis_values(
    result: dict[str, object],
    statistics_result: dict[str, object],
    metric: str,
    cluster: int,
) -> dict[str, list[float]]:
    settings = ClusteringStatisticsSettings.from_dict(
        dict(statistics_result.get("settings", {}))
    )
    groups = [str(value) for value in statistics_result.get("groups", [])]
    output: dict[str, list[float]] = {group: [] for group in groups}
    for group in groups:
        rows = [
            dict(row)
            for row in result.get("assignments", [])
            if str(row.get("experimental_group", "")) == group
            and int(row.get("morphology_cluster_id", 0)) == cluster
        ]
        if settings.analysis_level == "spine":
            output[group] = [
                value
                for value in (_metric_value(row, metric) for row in rows)
                if value is not None
            ]
            continue
        by_unit: dict[str, list[float]] = {}
        for row in rows:
            value = _metric_value(row, metric)
            if value is not None:
                by_unit.setdefault(
                    _unit_id(row, settings.analysis_level), []
                ).append(value)
        output[group] = [
            float(np.mean(values)) for values in by_unit.values() if values
        ]
    return output


def _protein_presence_values(
    result: dict[str, object],
    statistics_result: dict[str, object],
    cluster: int,
    *,
    aggregate_units: bool,
) -> dict[str, list[float]]:
    settings = ClusteringStatisticsSettings.from_dict(
        dict(statistics_result.get("settings", {}))
    )
    groups = [str(value) for value in statistics_result.get("groups", [])]
    output: dict[str, list[float]] = {group: [] for group in groups}
    for group in groups:
        rows = [
            dict(row)
            for row in result.get("assignments", [])
            if str(row.get("experimental_group", "")) == group
            and int(row.get("morphology_cluster_id", 0)) == cluster
        ]
        if not aggregate_units:
            output[group] = [
                float(bool(row.get("has_protein_cluster", False))) for row in rows
            ]
            continue
        by_unit: dict[str, list[float]] = {}
        for row in rows:
            by_unit.setdefault(
                _unit_id(row, settings.analysis_level), []
            ).append(float(bool(row.get("has_protein_cluster", False))))
        output[group] = [
            float(np.mean(values)) for values in by_unit.values() if values
        ]
    return output


def _draw_group_distributions(
    figure,
    grouped_values: dict[str, list[float]],
    comparison_rows: list[dict[str, object]],
    *,
    title: str,
    ylabel: str,
    color: str,
    percent: bool = False,
) -> None:
    axes = figure.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [3.2, 2.0]},
    )
    axis, comparison_axis = axes
    groups = list(grouped_values)
    positions = np.arange(1, len(groups) + 1, dtype=float)
    nonempty = [
        (position, np.asarray(grouped_values[group], dtype=np.float64))
        for position, group in zip(positions, groups)
        if grouped_values[group]
    ]
    if nonempty:
        artists = axis.boxplot(
            [values * (100.0 if percent else 1.0) for _position, values in nonempty],
            positions=[position for position, _values in nonempty],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
        )
        for box in artists["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.28)
        for position, values in nonempty:
            plotted = values * (100.0 if percent else 1.0)
            jitter = (
                np.linspace(-0.14, 0.14, len(plotted))
                if len(plotted) > 1
                else np.zeros(1)
            )
            axis.scatter(
                position + jitter,
                plotted,
                color=color,
                edgecolor="#202020",
                linewidth=0.35,
                alpha=0.78,
                s=24,
                zorder=3,
            )
    axis.set_xticks(positions, groups, rotation=25, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    if percent:
        axis.set_ylim(-3.0, 103.0)
    comparison_axis.set_axis_off()
    comparison_axis.set_title("Performed pairwise comparisons", loc="left")
    comparison_axis.text(
        0.0,
        0.97,
        "\n".join(_comparison_lines(comparison_rows)),
        ha="left",
        va="top",
        fontsize=8.5,
        wrap=True,
        transform=comparison_axis.transAxes,
    )


def draw_cluster_statistics(
    figure,
    result: dict[str, object],
    statistics_result: dict[str, object],
    mode: str,
    *,
    selected_metric: str | None = None,
    selected_cluster: int | None = None,
) -> None:
    """Draw every supported post-clustering statistical comparison."""

    figure.clear()
    composition = list(statistics_result.get("composition_summary", []))
    groups = list(statistics_result.get("groups", []))
    clusters = [int(value) for value in statistics_result.get("clusters", [])]
    definitions = list(result.get("cluster_definitions", []))
    colors = {
        int(row["morphology_cluster_id"]): str(row.get("color", "#457b9d"))
        for row in definitions
    }
    if not groups or not clusters:
        axis = figure.add_subplot(111)
        axis.text(0.5, 0.5, "No statistical plot data.", ha="center", va="center")
        return
    cluster = (
        int(selected_cluster)
        if selected_cluster is not None and int(selected_cluster) in clusters
        else clusters[0]
    )
    values = np.asarray(
        [
            [
                next(
                    (
                        float(row.get("percent_within_group") or 0.0)
                        for row in composition
                        if row.get("experimental_group") == group
                        and int(row.get("morphology_cluster_id", 0)) == cluster
                    ),
                    0.0,
                )
                for cluster in clusters
            ]
            for group in groups
        ],
        dtype=np.float64,
    )
    counts = np.asarray(
        [
            [
                next(
                    (
                        int(row.get("spine_count") or 0)
                        for row in composition
                        if row.get("experimental_group") == group
                        and int(row.get("morphology_cluster_id", 0)) == cluster
                    ),
                    0,
                )
                for cluster in clusters
            ]
            for group in groups
        ],
        dtype=np.int64,
    )
    if mode == "chi_square_residuals":
        axis = figure.add_subplot(111)
        if len(groups) < 2 or len(clusters) < 2:
            axis.text(
                0.5,
                0.5,
                "The chi-square residual plot requires at least two groups and two clusters.",
                ha="center",
                va="center",
                wrap=True,
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            return
        total_count = float(np.sum(counts))
        if total_count <= 0:
            axis.text(0.5, 0.5, "No counted spines were available.", ha="center", va="center")
            axis.set_axis_off()
            return
        expected = np.outer(np.sum(counts, axis=1), np.sum(counts, axis=0)) / total_count
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals = np.divide(
                counts - expected,
                np.sqrt(expected),
                out=np.zeros_like(expected, dtype=np.float64),
                where=expected > 0,
            )
        bound = max(2.0, float(np.nanmax(np.abs(residuals))))
        image = axis.imshow(
            residuals,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
        )
        for group_index in range(len(groups)):
            for cluster_index in range(len(clusters)):
                axis.text(
                    cluster_index,
                    group_index,
                    f"{residuals[group_index, cluster_index]:.2f}\n"
                    f"O={counts[group_index, cluster_index]}\n"
                    f"E={expected[group_index, cluster_index]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        axis.set_xticks(np.arange(len(clusters)), [f"Cluster {value}" for value in clusters])
        axis.set_yticks(np.arange(len(groups)), groups)
        omnibus = list(statistics_result.get("composition_omnibus", []))
        detail = (
            f"; p={float(omnibus[0]['p_value']):.3g}; "
            f"Cramer's V={float(omnibus[0]['cramers_v']):.3f}"
            if omnibus
            else ""
        )
        axis.set_title(f"Pearson residuals for group x cluster table{detail}")
        figure.colorbar(image, ax=axis, label="Pearson residual (observed - expected) / sqrt(expected)")
        return
    if mode == "unit_cluster_fractions":
        unit_rows = [
            row
            for row in statistics_result.get("unit_composition", [])
            if int(row.get("morphology_cluster_id", 0)) == cluster
        ]
        grouped = {
            group: [
                float(row["cluster_fraction"])
                for row in unit_rows
                if str(row.get("experimental_group", "")) == group
            ]
            for group in groups
        }
        comparisons = [
            row
            for row in statistics_result.get("unit_permutation", [])
            if int(row.get("morphology_cluster_id", 0)) == cluster
        ]
        _draw_group_distributions(
            figure,
            grouped,
            comparisons,
            title=f"Cluster {cluster} fractions across independent units",
            ylabel="Spines in cluster within unit (%)",
            color=colors.get(cluster, "#457b9d"),
            percent=True,
        )
        return
    if mode == "metric_distribution":
        available_metrics = list(
            dict.fromkeys(
                str(row.get("metric", ""))
                for row in statistics_result.get("metric_descriptives", [])
                if str(row.get("metric", ""))
            )
        )
        metric = (
            str(selected_metric)
            if selected_metric in available_metrics
            else (available_metrics[0] if available_metrics else "")
        )
        if not metric:
            axis = figure.add_subplot(111)
            axis.text(0.5, 0.5, "No continuous metrics were selected.", ha="center", va="center")
            axis.set_axis_off()
            return
        metric_rows = [
            row
            for row in statistics_result.get("metric_descriptives", [])
            if str(row.get("metric", "")) == metric
            and int(row.get("morphology_cluster_id", 0)) == cluster
        ]
        label = next(
            (str(row.get("metric_label", metric)) for row in metric_rows),
            metric,
        )
        comparisons = [
            row
            for row in statistics_result.get("metric_comparisons", [])
            if str(row.get("metric", "")) == metric
            and int(row.get("morphology_cluster_id", 0)) == cluster
        ]
        _draw_group_distributions(
            figure,
            _metric_analysis_values(result, statistics_result, metric, cluster),
            comparisons,
            title=f"{label} in morphology cluster {cluster}",
            ylabel=label,
            color=colors.get(cluster, "#457b9d"),
        )
        return
    if mode in {"protein_presence", "protein_presence_unit"}:
        aggregate = mode == "protein_presence_unit"
        grouped = _protein_presence_values(
            result,
            statistics_result,
            cluster,
            aggregate_units=aggregate,
        )
        key = (
            "protein_presence_unit_permutation"
            if aggregate
            else "protein_presence_comparisons"
        )
        comparisons = [
            row
            for row in statistics_result.get(key, [])
            if int(row.get("morphology_cluster_id", 0)) == cluster
        ]
        if aggregate:
            _draw_group_distributions(
                figure,
                grouped,
                comparisons,
                title=f"Protein-positive fraction per independent unit in cluster {cluster}",
                ylabel="Protein-positive spines within unit (%)",
                color=colors.get(cluster, "#457b9d"),
                percent=True,
            )
            return
        axes = figure.subplots(1, 2, gridspec_kw={"width_ratios": [3.2, 2.0]})
        axis, comparison_axis = axes
        percentages = [
            100.0 * float(np.mean(grouped[group])) if grouped[group] else 0.0
            for group in groups
        ]
        totals = [len(grouped[group]) for group in groups]
        axis.bar(groups, percentages, color=colors.get(cluster, "#457b9d"), alpha=0.8)
        for index, (percentage, total) in enumerate(zip(percentages, totals)):
            axis.text(index, percentage + 1.5, f"{percentage:.1f}%\n(n={total})", ha="center", va="bottom")
        axis.set_ylim(0, max(105.0, max(percentages, default=0.0) + 14.0))
        axis.set_ylabel("Protein-positive spines (%)")
        axis.set_title(f"Pooled protein-puncta presence in cluster {cluster}")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
        comparison_axis.set_axis_off()
        comparison_axis.set_title("Performed Agresti-Caffo comparisons", loc="left")
        comparison_axis.text(
            0.0,
            0.97,
            "\n".join(_comparison_lines(comparisons)),
            ha="left",
            va="top",
            fontsize=8.5,
            wrap=True,
            transform=comparison_axis.transAxes,
        )
        return
    if mode == "protein_presence_forest":
        comparisons = list(statistics_result.get("protein_presence_comparisons", []))
        axis = figure.add_subplot(111)
        if not comparisons:
            axis.text(0.5, 0.5, "No pooled protein-presence comparison was performed.", ha="center", va="center")
            axis.set_axis_off()
            return
        y = np.arange(len(comparisons))
        effects = np.asarray([float(row["adjusted_difference"]) * 100.0 for row in comparisons])
        low = np.asarray([float(row["ci_low"]) * 100.0 for row in comparisons])
        high = np.asarray([float(row["ci_high"]) * 100.0 for row in comparisons])
        axis.errorbar(effects, y, xerr=np.vstack((effects - low, high - effects)), fmt="none", ecolor="#555555", capsize=3)
        axis.scatter(
            effects,
            y,
            color=[colors.get(int(row["morphology_cluster_id"]), "#457b9d") for row in comparisons],
            zorder=3,
        )
        axis.axvline(0.0, color="#202020", linewidth=1.0)
        axis.set_yticks(
            y,
            [
                f"C{row['morphology_cluster_id']}: {row['first_group']} - {row['second_group']}"
                for row in comparisons
            ],
        )
        axis.set_xlabel("Adjusted protein-positive percentage-point difference")
        axis.set_title("Pairwise protein-puncta-presence differences")
        axis.grid(axis="x", alpha=0.2)
        return
    if mode == "faceted_pies":
        from matplotlib.patches import Patch

        columns = min(3, len(groups))
        rows = int(math.ceil(len(groups) / columns))
        axes = figure.subplots(rows, columns, squeeze=False)
        label_mode = str(statistics_result.get("settings", {}).get("pie_labels", "percent"))
        for index, group in enumerate(groups):
            axis = axes.flat[index]
            axis.pie(
                counts[index],
                colors=[colors.get(cluster, "#457b9d") for cluster in clusters],
                autopct=_pie_autopct(label_mode, int(np.sum(counts[index]))),
                startangle=90,
            )
            axis.set_title(f"{group} (n={int(np.sum(counts[index]))})")
        for axis in axes.flat[len(groups):]:
            axis.set_visible(False)
        figure.legend(
            handles=[
                Patch(
                    facecolor=colors.get(cluster, "#457b9d"),
                    label=f"Cluster {cluster}",
                )
                for cluster in clusters
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncols=min(len(clusters), 5),
        )
        figure.suptitle("Cluster composition within each experimental group")
        return
    axis = figure.add_subplot(111)
    if mode == "pooled_pie":
        pooled = np.sum(counts, axis=0)
        label_mode = str(statistics_result.get("settings", {}).get("pie_labels", "percent"))
        axis.pie(
            pooled,
            labels=[f"Cluster {cluster}" for cluster in clusters],
            colors=[colors.get(cluster, "#457b9d") for cluster in clusters],
            autopct=_pie_autopct(label_mode, int(np.sum(pooled))),
            startangle=90,
        )
        axis.set_title("Pooled cluster composition (all selected groups)")
    elif mode == "forest":
        comparisons = list(statistics_result.get("agresti_caffo", []))
        if not comparisons:
            axis.text(0.5, 0.5, "At least two groups are required.", ha="center", va="center")
            return
        y = np.arange(len(comparisons))
        effects = np.asarray([float(row["adjusted_difference"]) * 100.0 for row in comparisons])
        low = np.asarray([float(row["ci_low"]) * 100.0 for row in comparisons])
        high = np.asarray([float(row["ci_high"]) * 100.0 for row in comparisons])
        point_colors = [colors.get(int(row["morphology_cluster_id"]), "#457b9d") for row in comparisons]
        axis.errorbar(
            effects,
            y,
            xerr=np.vstack((effects - low, high - effects)),
            fmt="none",
            ecolor="#555555",
            capsize=3,
        )
        axis.scatter(effects, y, color=point_colors, zorder=3)
        axis.axvline(0.0, color="#202020", linewidth=1.0)
        axis.set_yticks(
            y,
            [
                f"C{row['morphology_cluster_id']}: {row['first_group']} - {row['second_group']}"
                for row in comparisons
            ],
        )
        axis.set_xlabel("Agresti-Caffo adjusted percentage-point difference")
        confidence = 100.0 * float(
            statistics_result.get("settings", {}).get("confidence_level", 0.95)
        )
        axis.set_title(
            f"Pairwise cluster-percentage differences ({confidence:g}% CI)"
        )
        axis.grid(axis="x", alpha=0.2)
    elif mode == "grouped":
        positions = np.arange(len(groups), dtype=float)
        width = 0.8 / max(1, len(clusters))
        for offset, cluster in enumerate(clusters):
            axis.bar(
                positions + (offset - (len(clusters) - 1) / 2.0) * width,
                values[:, offset],
                width=width,
                color=colors.get(cluster, "#457b9d"),
                label=f"Cluster {cluster}",
            )
        axis.set_xticks(positions, groups, rotation=25, ha="right")
        axis.set_ylabel("Spines within group (%)")
        axis.set_title("Cluster composition by experimental group")
        axis.legend()
    else:
        bottom = np.zeros(len(groups), dtype=np.float64)
        for cluster_index, cluster in enumerate(clusters):
            axis.bar(
                groups,
                values[:, cluster_index],
                bottom=bottom,
                color=colors.get(cluster, "#457b9d"),
                label=f"Cluster {cluster}",
            )
            bottom += values[:, cluster_index]
        axis.set_ylim(0, 100)
        axis.set_ylabel("Spines within group (%)")
        axis.set_title("100% stacked cluster composition")
        axis.tick_params(axis="x", rotation=25)
        axis.legend()


def statistical_methods_markdown(
    result: dict[str, object], statistics_result: dict[str, object] | None
) -> str:
    """Return an exact, human-readable audit of methods used for one run."""

    name = str(result.get("name", "Unnamed clustering run"))
    run_id = str(result.get("run_id", "standalone"))
    if not statistics_result:
        return (
            f"# Statistical methods for {name}\n\n"
            f"Run ID: `{run_id}`\n\n"
            "No inferential statistics were calculated for this run. The clustering "
            "export contains descriptive cluster summaries only.\n"
        )
    settings = ClusteringStatisticsSettings.from_dict(
        dict(statistics_result.get("settings", {}))
    )
    groups = ", ".join(str(value) for value in statistics_result.get("groups", []))
    input_features = set(result.get("settings", {}).get("features", []))
    descriptive_inputs = sorted(input_features & set(settings.metrics))
    warnings = list(statistics_result.get("warnings", []))
    level_definition = {
        "spine": "Each spine was treated as an independent observation.",
        "specimen": "Spines were aggregated within specimen; specimens received equal weight.",
        "neuron": "Specimens sharing the same neuron/subject ID were aggregated; neurons received equal weight.",
    }[settings.analysis_level]
    lines = [
        f"# Statistical methods for {name}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Analysis level: **{settings.analysis_level}**. {level_definition}",
        f"- Experimental groups: {groups or '(none)'}",
        f"- Confidence level: {settings.confidence_level * 100:.1f}%",
        f"- Monte Carlo repetitions: {settings.permutation_repetitions:,}",
        f"- Random seed: {settings.random_seed}",
        f"- Selected continuous-metric method: {settings.continuous_test}",
        "",
        "## Cluster composition",
        "",
        "A Pearson chi-square test compared the pooled group-by-cluster count table. "
        "Cramer's V is reported as its effect size. Expected counts below 5 are flagged; "
        "Synpo does not silently replace the requested test.",
        "",
        "For every experimental-group pair and every cluster, Synpo reports the "
        "Agresti-Caffo add-two-per-group interval and two-sided normal-approximation "
        "test for the difference in proportions. Effects are first selected group minus "
        "second selected group. These rows treat spines as independent.",
        "",
    ]
    if settings.analysis_level in {"specimen", "neuron"}:
        lines.extend(
            [
                f"At the selected {settings.analysis_level} level, cluster fractions were "
                "calculated inside each unit and compared by two-sided Monte Carlo label "
                "permutation of the difference in mean unit fractions.",
                "",
            ]
        )
    lines.extend(
        [
            "## Continuous and puncta metrics",
            "",
            "Continuous metrics were summarized inside each morphology cluster and "
            "experimental group. Spine mode uses individual values; specimen and neuron "
            "modes use unit means. Pairwise tests use the selected method, except neuron "
            "mode always uses Monte Carlo permutation as specified for this workflow.",
            "",
            "Metrics that were inputs to clustering are descriptive only because testing "
            "them after defining clusters with those same measurements is circular. "
            "Protein-puncta presence is also compared with Agresti-Caffo intervals; puncta "
            "count and volume are continuous metrics.",
            "",
            "Benjamini-Hochberg false-discovery-rate q-values are calculated separately "
            "for cluster-composition Agresti-Caffo tests, unit-level cluster-fraction "
            "permutations, continuous-metric comparisons, pooled protein-presence "
            "comparisons, and unit-level protein-presence permutations.",
            "",
            "Clustering-input metrics restricted to descriptive summaries: "
            + (", ".join(descriptive_inputs) if descriptive_inputs else "none"),
        ]
    )
    if warnings:
        lines.extend(["", "## Run warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return "\n".join(lines)
