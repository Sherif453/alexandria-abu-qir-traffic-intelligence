#!/usr/bin/env python3
"""Generate latest per-segment predictions from a trained model artifact."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from prediction_policy import (
  apply_decision_policy,
  build_feature_signals,
  count_decision_results,
  normalize_decision_policy,
  probabilities_to_label_map,
)


DEFAULT_FEATURE_VERSION = "features.v1"
DEFAULT_ARTIFACT_ROOT = "data/processed/models"
DEFAULT_MODEL_FILE = "random_forest.joblib"
DEFAULT_HORIZON_MINUTES = 15
FEATURE_COLUMNS = [
  "sort_order",
  "hour_of_day",
  "day_of_week",
  "weekend_flag",
  "holiday_flag",
  "recent_observed_speed",
  "rolling_mean_speed",
  "rolling_std_speed",
  "relative_to_free_flow",
  "speed_change_rate",
  "incident_flag",
  "segment_id",
  "road_category",
]


@dataclass(frozen=True)
class FeatureRecord:
  segment_id: str
  timestamp_utc: datetime
  sort_order: int
  hour_of_day: int | None
  day_of_week: int | None
  weekend_flag: bool | None
  holiday_flag: bool | None
  recent_observed_speed: float | None
  rolling_mean_speed: float | None
  rolling_std_speed: float | None
  relative_to_free_flow: float | None
  speed_change_rate: float | None
  incident_flag: bool | None
  road_category: str | None

  def to_feature_row(self) -> list[Any]:
    return [
      self.sort_order,
      self.hour_of_day,
      self.day_of_week,
      bool_to_int(self.weekend_flag),
      bool_to_int(self.holiday_flag),
      self.recent_observed_speed,
      self.rolling_mean_speed,
      self.rolling_std_speed,
      self.relative_to_free_flow,
      self.speed_change_rate,
      bool_to_int(self.incident_flag),
      self.segment_id,
      self.road_category,
    ]


@dataclass(frozen=True)
class PredictionWrite:
  segment_id: str
  timestamp_utc: datetime
  predicted_label: str
  confidence: float | None
  model_version: str


def bool_to_int(value: bool | None) -> int | None:
  if value is None:
    return None

  return 1 if value else 0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Generate latest predictions from FeatureSnapshot rows.",
  )
  parser.add_argument(
    "--database-url",
    default=os.environ.get("DATABASE_URL"),
    help="SQLite database URL. Defaults to DATABASE_URL from env or .env.",
  )
  parser.add_argument(
    "--feature-version",
    default=DEFAULT_FEATURE_VERSION,
    help=f"FeatureSnapshot version to score. Default: {DEFAULT_FEATURE_VERSION}.",
  )
  parser.add_argument(
    "--model-version",
    default=None,
    help="Model version to load. Defaults to latest ModelRun row.",
  )
  parser.add_argument(
    "--artifact-root",
    default=DEFAULT_ARTIFACT_ROOT,
    help=f"Model artifact root directory. Default: {DEFAULT_ARTIFACT_ROOT}.",
  )
  parser.add_argument(
    "--model-file",
    default=DEFAULT_MODEL_FILE,
    help=f"Model artifact filename. Default: {DEFAULT_MODEL_FILE}.",
  )
  parser.add_argument(
    "--horizon-minutes",
    type=int,
    default=DEFAULT_HORIZON_MINUTES,
    help=f"Forecast horizon used for the stored prediction timestamp. Default: {DEFAULT_HORIZON_MINUTES}.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Score and report predictions without writing Prediction rows.",
  )

  return parser.parse_args()


def import_joblib() -> Any:
  try:
    return importlib.import_module("joblib")
  except ImportError as error:
    raise SystemExit(
      "Missing Python ML dependencies. Run `npm run python:setup` first. "
      "On Ubuntu, install `python3.12-venv` if virtualenv creation fails."
    ) from error


def read_dotenv(root_dir: Path) -> dict[str, str]:
  env_path = root_dir / ".env"

  if not env_path.exists():
    return {}

  values: dict[str, str] = {}

  for line in env_path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()

    if not stripped or stripped.startswith("#") or "=" not in stripped:
      continue

    key, value = stripped.split("=", 1)
    values[key.strip()] = value.strip().strip("\"'")

  return values


def resolve_database_path(root_dir: Path, database_url: str | None) -> Path:
  resolved_url = database_url or read_dotenv(root_dir).get("DATABASE_URL")

  if not resolved_url:
    raise ValueError("DATABASE_URL is required. Set it in .env or pass --database-url.")

  if not resolved_url.startswith("file:"):
    raise ValueError("Only SQLite file: DATABASE_URL values are supported.")

  raw_path = resolved_url.removeprefix("file:")

  if raw_path.startswith("//"):
    raise ValueError("SQLite URI options are not supported by this predictor.")

  database_path = Path(raw_path)

  if database_path.is_absolute():
    return database_path

  return (root_dir / "prisma" / database_path).resolve()


def parse_timestamp(value: Any) -> datetime:
  if isinstance(value, datetime):
    timestamp = value
  elif isinstance(value, (int, float)):
    numeric = float(value)
    timestamp = datetime.fromtimestamp(
      numeric / 1000 if numeric > 10_000_000_000 else numeric,
      tz=UTC,
    )
  elif isinstance(value, str):
    normalized = value.strip()

    if normalized.endswith("Z"):
      normalized = f"{normalized[:-1]}+00:00"

    if normalized.endswith(" UTC"):
      normalized = normalized[:-4]

    timestamp = datetime.fromisoformat(normalized)
  else:
    raise TypeError(f"Unsupported timestamp value: {value!r}")

  if timestamp.tzinfo is None:
    return timestamp.replace(tzinfo=UTC)

  return timestamp.astimezone(UTC)


def format_timestamp(timestamp: datetime) -> str:
  return timestamp.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def to_optional_bool(value: Any) -> bool | None:
  if value is None:
    return None

  return bool(value)


def parse_model_metrics(metrics_json: str | None) -> dict[str, Any]:
  if not metrics_json:
    return {}

  try:
    parsed = json.loads(metrics_json)
  except json.JSONDecodeError:
    return {}

  return parsed if isinstance(parsed, dict) else {}


def get_model_classes(model: Any) -> list[str]:
  if hasattr(model, "classes_"):
    return list(model.classes_)

  classifier = getattr(model, "named_steps", {}).get("classifier")

  if classifier is not None and hasattr(classifier, "classes_"):
    return list(classifier.classes_)

  return []


def get_latest_model_run(
  connection: sqlite3.Connection,
  model_version: str | None,
) -> sqlite3.Row:
  if model_version:
    row = connection.execute(
      """
      SELECT "version", "artifactPath", "metricsJson", "createdAt"
      FROM "ModelRun"
      WHERE "version" = ?
      ORDER BY "createdAt" DESC
      LIMIT 1
      """,
      (model_version,),
    ).fetchone()
  else:
    row = connection.execute(
      """
      SELECT "version", "artifactPath", "metricsJson", "createdAt"
      FROM "ModelRun"
      ORDER BY "createdAt" DESC
      LIMIT 1
      """,
    ).fetchone()

  if row is None:
    raise ValueError("No ModelRun row found. Train a model before generating predictions.")

  if not row["artifactPath"]:
    raise ValueError(f"ModelRun {row['version']} does not have an artifact path.")

  return row


def load_latest_features(
  connection: sqlite3.Connection,
  feature_version: str,
) -> list[FeatureRecord]:
  rows = connection.execute(
    """
    WITH latest AS (
      SELECT
        "segmentId",
        MAX("timestampUtc") AS "latestTimestampUtc"
      FROM "FeatureSnapshot"
      WHERE "featureVersion" = ?
      GROUP BY "segmentId"
    )
    SELECT
      f."segmentId",
      f."timestampUtc",
      f."hourOfDay",
      f."dayOfWeek",
      f."weekendFlag",
      f."holidayFlag",
      f."recentObservedSpeed",
      f."rollingMeanSpeed",
      f."rollingStdSpeed",
      f."relativeToFreeFlow",
      f."speedChangeRate",
      f."incidentFlag",
      f."roadCategory",
      s."sortOrder"
    FROM "FeatureSnapshot" f
    JOIN latest l
      ON l."segmentId" = f."segmentId"
      AND l."latestTimestampUtc" = f."timestampUtc"
    JOIN "Segment" s ON s."segmentId" = f."segmentId"
    WHERE f."featureVersion" = ?
    ORDER BY s."sortOrder" ASC
    """,
    (feature_version, feature_version),
  ).fetchall()

  return [
    FeatureRecord(
      segment_id=row["segmentId"],
      timestamp_utc=parse_timestamp(row["timestampUtc"]),
      sort_order=row["sortOrder"],
      hour_of_day=row["hourOfDay"],
      day_of_week=row["dayOfWeek"],
      weekend_flag=to_optional_bool(row["weekendFlag"]),
      holiday_flag=to_optional_bool(row["holidayFlag"]),
      recent_observed_speed=row["recentObservedSpeed"],
      rolling_mean_speed=row["rollingMeanSpeed"],
      rolling_std_speed=row["rollingStdSpeed"],
      relative_to_free_flow=row["relativeToFreeFlow"],
      speed_change_rate=row["speedChangeRate"],
      incident_flag=to_optional_bool(row["incidentFlag"]),
      road_category=row["roadCategory"],
    )
    for row in rows
  ]


def predict(
  model: Any,
  features: list[FeatureRecord],
  model_version: str,
  horizon_minutes: int,
  decision_policy: dict[str, Any] | None,
  decisions: list[Any] | None = None,
  prediction_timestamp_utc: datetime | None = None,
) -> list[PredictionWrite]:
  feature_rows = [feature.to_feature_row() for feature in features]
  predicted_labels = model.predict(feature_rows).tolist()
  shared_prediction_timestamp_utc = prediction_timestamp_utc or (
    max(feature.timestamp_utc for feature in features) + timedelta(minutes=horizon_minutes)
  )
  predictions: list[PredictionWrite] = []

  for index, feature in enumerate(features):
    predicted_label = predicted_labels[index]
    confidence = None

    if decisions is not None:
      decision = decisions[index]
      predicted_label = decision.label
      confidence = float(decision.confidence) if decision.confidence is not None else None

    predictions.append(
      PredictionWrite(
        segment_id=feature.segment_id,
        timestamp_utc=shared_prediction_timestamp_utc,
        predicted_label=predicted_label,
        confidence=confidence,
        model_version=model_version,
      ),
    )

  return predictions


def build_prediction_decisions(
  model: Any,
  features: list[FeatureRecord],
  decision_policy: dict[str, Any] | None,
) -> list[Any] | None:
  if decision_policy is None or not hasattr(model, "predict_proba"):
    return None

  feature_rows = [feature.to_feature_row() for feature in features]
  probabilities = model.predict_proba(feature_rows)
  classes = get_model_classes(model)

  return [
    apply_decision_policy(
      probabilities=probabilities_to_label_map(classes, probabilities[index]),
      signals=build_feature_signals(
        relative_to_free_flow=feature.relative_to_free_flow,
        speed_change_rate=feature.speed_change_rate,
        incident_flag=feature.incident_flag,
      ),
      decision_policy=decision_policy,
    )
    for index, feature in enumerate(features)
  ]


def write_predictions(
  connection: sqlite3.Connection,
  predictions: list[PredictionWrite],
  model_version: str,
) -> int:
  if not predictions:
    return 0

  timestamp_strings = sorted({format_timestamp(prediction.timestamp_utc) for prediction in predictions})

  with connection:
    for timestamp_string in timestamp_strings:
      connection.execute(
        """
        DELETE FROM "Prediction"
        WHERE "modelVersion" = ?
          AND "timestampUtc" = ?
        """,
        (model_version, timestamp_string),
      )

    connection.executemany(
      """
      INSERT INTO "Prediction" (
        "id",
        "segmentId",
        "timestampUtc",
        "predictedLabel",
        "confidence",
        "modelVersion",
        "createdAt"
      ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
      """,
      [
        (
          f"prediction_{uuid.uuid4().hex}",
          prediction.segment_id,
          format_timestamp(prediction.timestamp_utc),
          prediction.predicted_label,
          prediction.confidence,
          prediction.model_version,
        )
        for prediction in predictions
      ],
    )

  return len(predictions)


def summarize_predictions(predictions: list[PredictionWrite]) -> dict[str, int]:
  counts: dict[str, int] = {}

  for prediction in predictions:
    counts[prediction.predicted_label] = counts.get(prediction.predicted_label, 0) + 1

  return dict(sorted(counts.items()))


def main() -> None:
  args = parse_args()
  root_dir = Path(__file__).resolve().parents[2]
  database_path = resolve_database_path(root_dir, args.database_url)

  if args.horizon_minutes < 1:
    raise ValueError("--horizon-minutes must be at least 1.")

  if not database_path.exists():
    raise FileNotFoundError(f"SQLite database not found: {database_path}")

  connection = sqlite3.connect(database_path)
  connection.row_factory = sqlite3.Row

  try:
    model_run = get_latest_model_run(connection, args.model_version)
    model_version = model_run["version"]
    artifact_path = Path(model_run["artifactPath"])
    model_path = (root_dir / artifact_path / args.model_file).resolve()
    model_metrics = parse_model_metrics(model_run["metricsJson"])
    serving_policy = model_metrics.get("servingPolicy")
    decision_policy = None

    if isinstance(serving_policy, dict):
      decision_policy = normalize_decision_policy(serving_policy.get("decisionPolicy"))

    if not model_path.exists():
      raise FileNotFoundError(f"Model artifact not found: {model_path}")

    features = load_latest_features(connection, args.feature_version)

    if not features:
      raise ValueError("No FeatureSnapshot rows found. Build features before generating predictions.")

    joblib = import_joblib()
    model = joblib.load(model_path)
    decisions = build_prediction_decisions(model, features, decision_policy)
    prediction_timestamp_utc = max(feature.timestamp_utc for feature in features) + timedelta(
      minutes=args.horizon_minutes,
    )
    predictions = predict(
      model=model,
      features=features,
      model_version=model_version,
      horizon_minutes=args.horizon_minutes,
      decision_policy=decision_policy,
      decisions=decisions,
      prediction_timestamp_utc=prediction_timestamp_utc,
    )
    written = 0 if args.dry_run else write_predictions(connection, predictions, model_version)
    feature_timestamps = [feature.timestamp_utc for feature in features]
    prediction_timestamps = [prediction.timestamp_utc for prediction in predictions]
    decision_counts = {"labels": summarize_predictions(predictions)}

    if decisions is not None:
      decision_counts = count_decision_results(decisions)

    print(
      json.dumps(
        {
          "status": "ok",
          "dryRun": args.dry_run,
          "databasePath": str(database_path),
          "modelVersion": model_version,
          "modelPath": str(model_path.relative_to(root_dir)),
          "featureVersion": args.feature_version,
          "horizonMinutes": args.horizon_minutes,
          "decisionPolicy": decision_policy,
          "featuresScored": len(features),
          "predictionsWritten": written,
          "predictionCounts": summarize_predictions(predictions),
          "decisionCounts": decision_counts,
          "featureRange": {
            "fromUtc": format_timestamp(min(feature_timestamps)),
            "toUtc": format_timestamp(max(feature_timestamps)),
          },
          "predictionRange": {
            "fromUtc": format_timestamp(min(prediction_timestamps)),
            "toUtc": format_timestamp(max(prediction_timestamps)),
          },
        },
        indent=2,
      ),
    )
  finally:
    connection.close()


if __name__ == "__main__":
  main()
