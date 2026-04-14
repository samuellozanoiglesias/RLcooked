#!/usr/bin/env python3
"""
Episode-focused ability analysis for classic RL experiments.

The script loads one map/configuration, filters the training data by a set of
hardcoded ability configurations, and summarizes the classic movement and
meaningful-action metrics around a target episode.

Instead of plotting episode curves, it computes the mean over a window centered
on the target episode and renders grouped bar charts for each ability
configuration.

Example:
    nohup python indepth_abilities_analysis.py encouraged_division_of_labor_large --target_episode=750 --window_size=20 --specialization 0.05 --synergy 1.70 > indepth_abilities_analysis.out 2>&1 &
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spoiled_broth.analysis.utils import MetricDefinitions, main_analysis_pipeline


@dataclass(frozen=True)
class AbilityConfig:
	config_id: str
	label: str
	walk1: float
	cut1: float
	walk2: float
	cut2: float


ABILITY_CONFIGS: List[AbilityConfig] = [
	AbilityConfig("asym_1p0", "A1 1.0/1.0 | A2 1.0/0.3", 1.0, 1.0, 1.0, 0.3),
	AbilityConfig("asym_0p9", "A1 0.9/1.0 | A2 1.0/0.3", 0.9, 1.0, 1.0, 0.3),
	AbilityConfig("asym_0p8", "A1 0.8/1.0 | A2 1.0/0.3", 0.8, 1.0, 1.0, 0.3),
	AbilityConfig("asym_0p7", "A1 0.7/1.0 | A2 1.0/0.3", 0.7, 1.0, 1.0, 0.3),
	AbilityConfig("asym_0p6", "A1 0.6/1.0 | A2 1.0/0.3", 0.6, 1.0, 1.0, 0.3),
	AbilityConfig("asym_0p5", "A1 0.5/1.0 | A2 1.0/0.3", 0.5, 1.0, 1.0, 0.3),
	AbilityConfig("asym_0p4", "A1 0.4/1.0 | A2 1.0/0.3", 0.4, 1.0, 1.0, 0.3),
	AbilityConfig("asym_0p3", "A1 0.3/1.0 | A2 1.0/0.3", 0.3, 1.0, 1.0, 0.3),
	AbilityConfig("asym_0p2", "A1 0.2/1.0 | A2 1.0/0.3", 0.2, 1.0, 1.0, 0.3),
	AbilityConfig("asym_0p1", "A1 0.1/1.0 | A2 1.0/0.3", 0.1, 1.0, 1.0, 0.3),
]


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Episode-window ability analysis for classic experiments",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("map_name", type=str, help="Map name identifier")
	parser.add_argument("--cluster", type=str, choices=["brigit", "cuenca", "local"], default="brigit")
	parser.add_argument("--study_name", type=str, default=None)
	parser.add_argument("--game_type", type=str, choices=["classic", "classic_collision"], default="classic_collision")
	parser.add_argument("--init_type", type=str, choices=["random_init", "empty_init"], default="empty_init")
	parser.add_argument("--synergy", type=float, default=None, help="Synergy scaling factor")
	parser.add_argument("--specialization", type=float, default=None, help="Specialization lambda")
	parser.add_argument("--target_episode", type=int, required=True, help="Nth episode within each training run")
	parser.add_argument("--window_size", type=int, default=20, help="Episode neighborhood size around target episode")
	parser.add_argument(
		"--ability_configs",
		type=str,
		default=None,
		help="Optional comma-separated subset of hardcoded ability config ids",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="Optional custom output directory. Defaults to the experiment figures directory.",
	)
	return parser


def _selected_ability_configs(selection: Optional[str]) -> List[AbilityConfig]:
	if not selection:
		return ABILITY_CONFIGS

	wanted = {item.strip() for item in selection.split(",") if item.strip()}
	selected = [config for config in ABILITY_CONFIGS if config.config_id in wanted]
	missing = sorted(wanted.difference({config.config_id for config in selected}))
	if missing:
		raise ValueError(f"Unknown ability config id(s): {', '.join(missing)}")
	return selected


def _sanitize_token(value: str) -> str:
	return value.replace("/", "_").replace(" ", "_").replace(".", "p").replace("|", "_")


def _default_output_dir(raw_dir: str) -> str:
	return os.path.join(raw_dir, "indepth_abilities_analysis_figures")


def _chunked(values: Sequence[str], size: int) -> List[List[str]]:
	return [list(values[index:index + size]) for index in range(0, len(values), size)]


def _filter_by_ability_config(df: pd.DataFrame, ability: AbilityConfig) -> pd.DataFrame:
	required_cols = ["walking_speed_1", "cutting_speed_1", "walking_speed_2", "cutting_speed_2"]
	if not all(col in df.columns for col in required_cols):
		return pd.DataFrame()

	condition = (
		(df["walking_speed_1"].sub(ability.walk1).abs() < 0.01)
		& (df["cutting_speed_1"].sub(ability.cut1).abs() < 0.01)
		& (df["walking_speed_2"].sub(ability.walk2).abs() < 0.01)
		& (df["cutting_speed_2"].sub(ability.cut2).abs() < 0.01)
	)
	return df.loc[condition].copy()


def _window_slice(training_df: pd.DataFrame, target_episode: int, window_size: int) -> Tuple[pd.DataFrame, int, Tuple[int, int]]:
	episodes = sorted(training_df["episode"].dropna().astype(int).unique())
	if not episodes:
		return training_df.iloc[0:0].copy(), target_episode, (target_episode, target_episode)

	if target_episode <= 0:
		raise ValueError("target_episode must be a positive integer")

	if target_episode > len(episodes):
		target_episode = len(episodes)

	actual_target_episode = int(episodes[target_episode - 1])
	half_window = max(window_size // 2, 0)
	start_episode = actual_target_episode - half_window
	end_episode = actual_target_episode + half_window

	window_df = training_df.loc[(training_df["episode"] >= start_episode) & (training_df["episode"] <= end_episode)].copy()
	return window_df, actual_target_episode, (start_episode, end_episode)


def _summarize_training_window(
	training_df: pd.DataFrame,
	target_episode: int,
	window_size: int,
	metric_columns: Sequence[str],
) -> Tuple[pd.Series, int, Tuple[int, int], int]:
	window_df, actual_target_episode, bounds = _window_slice(training_df, target_episode, window_size)
	if window_df.empty:
		return pd.Series(dtype=float), actual_target_episode, bounds, 0

	present_columns = [col for col in metric_columns if col in window_df.columns]
	if not present_columns:
		return pd.Series(dtype=float), actual_target_episode, bounds, len(window_df)

	return window_df[present_columns].mean(), actual_target_episode, bounds, len(window_df)


def _combine_training_windows(
	filtered_df: pd.DataFrame,
	target_episode: int,
	window_size: int,
	metric_columns: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
	rows: List[pd.Series] = []
	metadata: Dict[str, object] = {
		"training_count": 0,
		"window_rows": 0,
		"actual_target_episode": None,
		"episode_start": None,
		"episode_end": None,
	}

	for timestamp, training_df in filtered_df.groupby("timestamp", sort=True):
		if training_df.empty:
			continue

		window_mean, actual_target_episode, bounds, window_rows = _summarize_training_window(
			training_df,
			target_episode,
			window_size,
			metric_columns,
		)

		if window_mean.empty:
			continue

		window_mean["timestamp"] = timestamp
		rows.append(window_mean)
		metadata["training_count"] += 1
		metadata["window_rows"] += window_rows
		metadata["actual_target_episode"] = actual_target_episode
		metadata["episode_start"] = bounds[0]
		metadata["episode_end"] = bounds[1]

	if not rows:
		return pd.DataFrame(), metadata

	combined = pd.DataFrame(rows)
	value_columns = [col for col in metric_columns if col in combined.columns]
	summary = pd.DataFrame(
		{
			"mean": combined[value_columns].mean(),
			"std": combined[value_columns].std(ddof=0),
			"n_trainings": metadata["training_count"],
			"window_rows": metadata["window_rows"],
			"target_episode": target_episode,
			"actual_target_episode": metadata["actual_target_episode"],
			"episode_start": metadata["episode_start"],
			"episode_end": metadata["episode_end"],
		}
	).reset_index(names="metric")
	return summary, metadata


def _family_summary(summary: pd.DataFrame, base_metrics: Sequence[str]) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []
	for base_metric in base_metrics:
		row: Dict[str, object] = {"metric": base_metric}
		for agent_num in (1, 2):
			metric_name = f"{base_metric}_ai_rl_{agent_num}"
			metric_row = summary.loc[summary["metric"] == metric_name]
			if metric_row.empty:
				row[f"agent_{agent_num}_mean"] = np.nan
				row[f"agent_{agent_num}_std"] = np.nan
				continue
			row[f"agent_{agent_num}_mean"] = float(metric_row["mean"].iloc[0])
			row[f"agent_{agent_num}_std"] = float(metric_row["std"].iloc[0]) if not pd.isna(metric_row["std"].iloc[0]) else 0.0
		rows.append(row)

	family_df = pd.DataFrame(rows)
	if not family_df.empty:
		first_row = summary.iloc[0]
		for column in ("n_trainings", "target_episode", "actual_target_episode", "episode_start", "episode_end"):
			family_df[column] = first_row[column]
	return family_df


def _process_ability_config(
	df: pd.DataFrame,
	ability: AbilityConfig,
	target_episode: int,
	window_size: int,
) -> List[Dict[str, object]]:
	base_metrics = MetricDefinitions.get_base_classic_metrics()
	meaningful_actions = base_metrics["rewarded_metrics"]
	movement_metrics = base_metrics["movement_metrics"]
	all_base_metrics = meaningful_actions + movement_metrics
	metric_columns = [f"{metric}_ai_rl_{agent_num}" for metric in all_base_metrics for agent_num in (1, 2)]

	filtered = _filter_by_ability_config(df, ability)
	if filtered.empty:
		print(f"  Skipping {ability.config_id}: no rows matched the hardcoded ability tuple")
		return []

	summary, metadata = _combine_training_windows(filtered, target_episode, window_size, metric_columns)
	if summary.empty:
		print(f"  Skipping {ability.config_id}: no metrics found in the target window")
		return []

	actual_target_episode = int(metadata["actual_target_episode"] or target_episode)
	meaningful_df = _family_summary(summary, meaningful_actions)
	movement_df = _family_summary(summary, movement_metrics)

	rows: List[Dict[str, object]] = []
	for family_name, family_df in (("meaningful_actions", meaningful_df), ("movement_metrics", movement_df)):
		for _, row in family_df.iterrows():
			for agent_num in (1, 2):
				rows.append(
					{
						"ability_config": ability.config_id,
						"ability_label": ability.label,
						"family": family_name,
						"metric": row["metric"],
						"agent": agent_num,
						"mean": row[f"agent_{agent_num}_mean"],
						"std": row[f"agent_{agent_num}_std"],
						"n_trainings": int(row.get("n_trainings", 0)),
						"target_episode": target_episode,
						"actual_target_episode": actual_target_episode,
						"window_size": window_size,
					}
				)

	return rows


def _plot_family_comparison(
	comparison_df: pd.DataFrame,
	family_name: str,
	ability_order: Sequence[AbilityConfig],
	output_dir: str,
	title_prefix: str,
	target_episode: int,
	window_size: int,
	metric_order: Sequence[str],
) -> None:
	if comparison_df.empty:
		return

	metric_labels = MetricDefinitions.get_metric_labels()
	ability_labels = [ability.label for ability in ability_order]
	ability_ids = [ability.config_id for ability in ability_order]
	family_subset = comparison_df.loc[comparison_df["family"] == family_name].copy()
	if family_subset.empty:
		return

	family_subset["ability_config"] = pd.Categorical(family_subset["ability_config"], categories=ability_ids, ordered=True)
	family_subset = family_subset.sort_values(["metric", "ability_config", "agent"])

	metric_list = [metric for metric in metric_order if metric in set(family_subset["metric"])]
	if not metric_list:
		return

	cols = 3 if len(metric_list) >= 6 else 2
	rows = int(np.ceil(len(metric_list) / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(max(14, cols * 5), max(8, rows * 3.2)), sharex=True)
	axes_flat = np.atleast_1d(axes).ravel()
	width = 0.35
	bar_offsets = {1: -width / 2, 2: width / 2}
	bar_colors = {1: "#2E86DE", 2: "#E74C3C"}

	for axis_index, metric in enumerate(metric_list):
		axis = axes_flat[axis_index]
		metric_df = family_subset.loc[family_subset["metric"] == metric]
		metric_df = metric_df.set_index(["ability_config", "agent"])
		x_positions = np.arange(len(ability_ids))
		for agent_num in (1, 2):
			means = []
			errors = []
			for ability_id in ability_ids:
				if (ability_id, agent_num) in metric_df.index:
					row = metric_df.loc[(ability_id, agent_num)]
					means.append(float(row["mean"]))
					errors.append(float(row["std"]) if not pd.isna(row["std"]) else 0.0)
				else:
					means.append(np.nan)
					errors.append(0.0)
			axis.bar(
				x_positions + bar_offsets[agent_num],
				means,
				width=width,
				label=f"Agent {agent_num}" if axis_index == 0 else None,
				color=bar_colors[agent_num],
				edgecolor="#1f1f1f",
				linewidth=0.8,
				yerr=errors,
				capsize=3,
			)
		axis.set_title(metric_labels.get(metric, metric.replace("_", " ").title()), fontsize=11)
		axis.grid(axis="y", alpha=0.25)
		axis.set_xticks(x_positions)
		if axis_index >= len(metric_list) - cols:
			axis.set_xticklabels(ability_labels, rotation=25, ha="right")
		else:
			axis.set_xticklabels([])
		axis.tick_params(axis="x", length=0)

	for axis in axes_flat[len(metric_list):]:
		axis.axis("off")

	fig.suptitle(
		f"{title_prefix}\n{family_name} | target episode {target_episode} | window {window_size} | abilities compared {len(ability_ids)}",
		fontsize=14,
	)
	fig.legend(loc="upper right", bbox_to_anchor=(0.985, 0.985))
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	fig.savefig(os.path.join(output_dir, f"{family_name}_comparison_target{target_episode}_window{window_size}.png"), dpi=300, bbox_inches="tight")
	plt.close(fig)


def _write_comparison_summary(comparison_df: pd.DataFrame, output_dir: str, target_episode: int, window_size: int) -> None:
	if comparison_df.empty:
		return

	comparison_df.to_csv(
		os.path.join(output_dir, f"indepth_abilities_summary_target{target_episode}_window{window_size}.csv"),
		index=False,
	)


def main() -> None:
	parser = _build_parser()
	args = parser.parse_args()
	ability_configs = _selected_ability_configs(args.ability_configs)

	print("Starting in-depth ability analysis...")
	print(f"Map: {args.map_name}")
	print(f"Cluster: {args.cluster}")
	print(f"Game type: {args.game_type}")
	print(f"Init type: {args.init_type}")
	print(f"Target episode: {args.target_episode}")
	print(f"Window size: {args.window_size}")
	if args.synergy is not None:
		print(f"Synergy: {args.synergy}")
	if args.specialization is not None:
		print(f"Specialization: {args.specialization}")
	print(f"Ability configs: {[ability.config_id for ability in ability_configs]}")

	try:
		analysis_results = main_analysis_pipeline(
			experiment_type="classic",
			map_name=args.map_name,
			cluster=args.cluster,
			study_name=args.study_name,
			game_type=args.game_type,
			init_type=args.init_type,
			synergy_scaling_factor=args.synergy if args.synergy is not None else 0.0,
			synergy_provided=args.synergy is not None,
			specialization_lambda=args.specialization,
		)

		if isinstance(analysis_results, dict) and "df" not in analysis_results:
			result_items = analysis_results.items()
		else:
			result_items = [("single_result", analysis_results)]

		combined_rows: List[Dict[str, object]] = []
		multiple_results = len(result_items) > 1
		base_output_dir = args.output_dir

		for result_key, result in result_items:
			df = result["df"]
			paths = result["paths"]
			if base_output_dir is None:
				base_output_dir = _default_output_dir(paths["raw_dir"])
			os.makedirs(base_output_dir, exist_ok=True)
			output_dir = base_output_dir
			if multiple_results:
				output_dir = os.path.join(base_output_dir, _sanitize_token(result_key))
			os.makedirs(output_dir, exist_ok=True)

			title_bits = [f"Map {args.map_name}", f"{args.game_type}"]
			if args.synergy is not None:
				title_bits.append(f"synergy={args.synergy}")
			if args.specialization is not None:
				title_bits.append(f"specialization={args.specialization}")
			title_prefix = " | ".join(title_bits)

			print(f"\nProcessing result set: {result_key}")
			for ability in ability_configs:
				print(f"  Ability config: {ability.config_id}")
				rows = _process_ability_config(
					df=df,
					ability=ability,
					target_episode=args.target_episode,
					window_size=args.window_size,
				)
				combined_rows.extend(rows)

		if combined_rows:
			summary_df = pd.DataFrame(combined_rows)
			summary_base_dir = base_output_dir
			if "ability_config" in summary_df.columns:
				comparison_df = summary_df.loc[summary_df["ability_config"].notna()].copy()
				comparison_df = comparison_df.loc[comparison_df["family"].isin(["meaningful_actions", "movement_metrics"])]
				_write_comparison_summary(comparison_df, summary_base_dir, args.target_episode, args.window_size)

			_plot_family_comparison(
				summary_df,
				"meaningful_actions",
				ability_configs,
				summary_base_dir,
				title_prefix,
				args.target_episode,
				args.window_size,
				MetricDefinitions.get_base_classic_metrics()["rewarded_metrics"],
			)
			_plot_family_comparison(
				summary_df,
				"movement_metrics",
				ability_configs,
				summary_base_dir,
				title_prefix,
				args.target_episode,
				args.window_size,
				MetricDefinitions.get_base_classic_metrics()["movement_metrics"],
			)

			print(f"Saved comparison figures to {summary_base_dir}")

		print("In-depth ability analysis completed successfully.")

	except Exception as exc:
		print(f"Error during in-depth ability analysis: {exc}")
		sys.exit(1)


if __name__ == "__main__":
	main()
