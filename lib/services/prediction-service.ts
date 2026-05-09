import {
  CORRIDOR_DEFINITION_VERSION,
  CORRIDOR_ID,
  CORRIDOR_NAME,
  CORRIDOR_SCOPE,
} from "@/lib/corridor/definition";
import {
  getLatestFeatureSnapshotsForSegments,
  getRecentFeatureSnapshotsForSegments,
} from "@/lib/repositories/feature-snapshot-repository";
import {
  getLatestModelRun,
  getModelRunByVersion,
} from "@/lib/repositories/model-run-repository";
import {
  getPredictionsForSegmentsInCreatedBatch,
  getPredictionsForSegmentsAtTimestamp,
  listPredictionSnapshotCandidates,
  type PredictionSnapshotCandidate,
} from "@/lib/repositories/prediction-repository";
import { listSegments } from "@/lib/repositories/segment-repository";
import { getLatestTrafficObservations } from "@/lib/repositories/traffic-observation-repository";
import { syncSegmentsFromDefinition } from "@/lib/services/segment-service";
import {
  getLiveWindowPayload,
  getWindowAwareFreshnessStatus,
  type FreshnessStatus,
  type LiveWindowPayload,
} from "@/lib/time/live-window";

type LatestObservation = Awaited<ReturnType<typeof getLatestTrafficObservations>>[number];
type LatestPrediction = Awaited<ReturnType<typeof getPredictionsForSegmentsAtTimestamp>>[number];
type LatestFeature = Awaited<ReturnType<typeof getLatestFeatureSnapshotsForSegments>>[number];

const FEATURE_VERSION = "features.v1";
const PREDICTION_FRESHNESS_MINUTES = 30;
const PREDICTION_HORIZON_MINUTES = 15;

const severityScore: Record<string, number> = {
  Low: 0,
  Medium: 1,
  High: 2,
};

function formatCongestionForReason(label: string | null | undefined): string {
  if (label === "Low") {
    return "low congestion";
  }

  if (label === "Medium") {
    return "medium congestion";
  }

  if (label === "High") {
    return "high congestion";
  }

  return "unknown congestion";
}

function toIsoString(date: Date | null | undefined): string | null {
  return date ? date.toISOString() : null;
}

function average(values: Array<number | null | undefined>): number | null {
  const usableValues = values.filter(
    (value): value is number => typeof value === "number" && Number.isFinite(value),
  );

  if (usableValues.length === 0) {
    return null;
  }

  return usableValues.reduce((sum, value) => sum + value, 0) / usableValues.length;
}

function countLabels(records: Array<{ predictedLabel?: string; congestionLabel?: string | null }>) {
  return records.reduce<Record<string, number>>((counts, record) => {
    const label = record.predictedLabel ?? record.congestionLabel ?? "Unknown";
    counts[label] = (counts[label] ?? 0) + 1;

    return counts;
  }, {});
}

function buildCorridorPayload(segmentCount: number) {
  return {
    id: CORRIDOR_ID,
    name: CORRIDOR_NAME,
    scope: CORRIDOR_SCOPE,
    definitionVersion: CORRIDOR_DEFINITION_VERSION,
    samplePointCount: segmentCount,
  };
}

function mapObservation(observation: NonNullable<LatestObservation>) {
  const speedRatio =
    observation.speed !== null &&
    observation.freeFlowSpeed !== null &&
    observation.freeFlowSpeed > 0
      ? observation.speed / observation.freeFlowSpeed
      : null;

  return {
    id: observation.id,
    segmentId: observation.segmentId,
    timestampUtc: observation.timestampUtc.toISOString(),
    speed: observation.speed,
    freeFlowSpeed: observation.freeFlowSpeed,
    speedRatio,
    congestionLabel: observation.congestionLabel,
    source: observation.source,
    qualityStatus: observation.qualityStatus,
    ingestionRunId: observation.ingestionRunId,
  };
}

function mapPrediction(prediction: NonNullable<LatestPrediction>) {
  return {
    id: prediction.id,
    segmentId: prediction.segmentId,
    timestampUtc: prediction.timestampUtc.toISOString(),
    predictedLabel: prediction.predictedLabel,
    confidence: prediction.confidence,
    modelVersion: prediction.modelVersion,
    createdAt: prediction.createdAt.toISOString(),
  };
}

function parseModelWarnings(metricsJson: string | null | undefined): string[] {
  if (!metricsJson) {
    return [];
  }

  try {
    const metrics = JSON.parse(metricsJson) as { warnings?: unknown };

    if (Array.isArray(metrics.warnings)) {
      return metrics.warnings.filter((warning): warning is string => typeof warning === "string");
    }
  } catch {
    return [];
  }

  return [];
}

function getFeatureTimestampForPrediction(predictionTimestampUtc: Date): Date {
  return new Date(
    predictionTimestampUtc.getTime() - PREDICTION_HORIZON_MINUTES * 60 * 1000,
  );
}

export function selectBestPredictionSnapshot(
  candidates: PredictionSnapshotCandidate[],
): PredictionSnapshotCandidate | null {
  if (candidates.length === 0) {
    return null;
  }

  return [...candidates].sort((left, right) => {
    if (right.predictedSegments !== left.predictedSegments) {
      return right.predictedSegments - left.predictedSegments;
    }

    return right.timestampUtc.getTime() - left.timestampUtc.getTime();
  })[0] ?? null;
}

async function resolvePredictionSnapshot(params: {
  segmentIds: string[];
  preferredModelVersion?: string | null;
}) {
  const preferredCandidates = params.preferredModelVersion
    ? await listPredictionSnapshotCandidates({
        segmentIds: params.segmentIds,
        modelVersion: params.preferredModelVersion,
        take: 12,
      })
    : [];
  const preferredSnapshot = selectBestPredictionSnapshot(preferredCandidates);

  if (preferredSnapshot) {
    return preferredSnapshot;
  }

  const fallbackCandidates = await listPredictionSnapshotCandidates({
    segmentIds: params.segmentIds,
    take: 12,
  });

  return selectBestPredictionSnapshot(fallbackCandidates);
}

function getPredictionFreshness(
  predictions: Array<LatestPrediction>,
  latestPredictionTimestampUtc: Date | null,
  liveWindow: LiveWindowPayload,
): FreshnessStatus {
  const predictionRows = predictions.filter(
    (prediction): prediction is NonNullable<typeof prediction> => Boolean(prediction),
  );

  if (predictionRows.length === 0) {
    return "empty";
  }

  return getWindowAwareFreshnessStatus({
    latestTimestampUtc: latestPredictionTimestampUtc,
    liveWindow,
    checkedAtUtc: new Date(liveWindow.checkedAtUtc),
    freshForMinutes: PREDICTION_FRESHNESS_MINUTES,
  });
}

function getTrend(params: {
  feature: LatestFeature;
  prediction: LatestPrediction;
  observation: LatestObservation;
}): {
  trend: "improving" | "stable" | "worsening" | "uncertain";
  reason: string;
} {
  if (!params.observation || !params.prediction || !params.observation.congestionLabel) {
    return {
      trend: "uncertain",
      reason: "There is not enough recent information for this area.",
    };
  }

  const currentScore = severityScore[params.observation.congestionLabel];
  const predictedScore = severityScore[params.prediction.predictedLabel];

  if (currentScore === undefined || predictedScore === undefined) {
    return {
      trend: "uncertain",
      reason: "This area could not be placed into a clear congestion level.",
    };
  }

  if (predictedScore > currentScore) {
    return {
      trend: "worsening",
      reason: `Congestion is expected to increase from ${formatCongestionForReason(
        params.observation.congestionLabel,
      )} to ${formatCongestionForReason(params.prediction.predictedLabel)}.`,
    };
  }

  if (predictedScore < currentScore) {
    return {
      trend: "improving",
      reason: `Congestion is expected to decrease from ${formatCongestionForReason(
        params.observation.congestionLabel,
      )} to ${formatCongestionForReason(params.prediction.predictedLabel)}.`,
    };
  }

  if (params.feature?.speedChangeRate !== null && params.feature?.speedChangeRate !== undefined) {
    if (params.feature.speedChangeRate < -0.15) {
      return {
        trend: "stable",
        reason: `Congestion is expected to remain ${formatCongestionForReason(
          params.prediction.predictedLabel,
        )}. Recent speed is falling, so this area needs a closer look.`,
      };
    }

    if (params.feature.speedChangeRate > 0.15) {
      return {
        trend: "stable",
        reason: `Congestion is expected to remain ${formatCongestionForReason(
          params.prediction.predictedLabel,
        )}. Recent speed is rising, but the congestion level is unchanged.`,
      };
    }
  }

  return {
    trend: "stable",
    reason: `Congestion is expected to remain ${formatCongestionForReason(
      params.prediction.predictedLabel,
    )}.`,
  };
}

export async function getLatestPredictionsPayload() {
  await syncSegmentsFromDefinition();

  const segments = await listSegments();
  const segmentIds = segments.map((segment) => segment.segmentId);
  const latestModelRun = await getLatestModelRun();
  const latestObservations = await getLatestTrafficObservations(segmentIds);
  const observationsBySegmentId = new Map(
    latestObservations
      .filter((observation): observation is NonNullable<typeof observation> =>
        Boolean(observation),
      )
      .map((observation) => [observation.segmentId, observation]),
  );
  const snapshot = await resolvePredictionSnapshot({
    segmentIds,
    preferredModelVersion: latestModelRun?.version ?? null,
  });
  const modelRun = snapshot
    ? snapshot.modelVersion === latestModelRun?.version
      ? latestModelRun
      : await getModelRunByVersion(snapshot.modelVersion)
    : latestModelRun;
  const latestPredictionTimestampUtc = snapshot?.timestampUtc ?? null;
  const predictions =
    snapshot && latestPredictionTimestampUtc
      ? snapshot.createdAt
        ? await getPredictionsForSegmentsInCreatedBatch({
            segmentIds,
            modelVersion: snapshot.modelVersion,
            createdAt: snapshot.createdAt,
          })
        : await getPredictionsForSegmentsAtTimestamp({
            segmentIds,
            modelVersion: snapshot.modelVersion,
            timestampUtc: latestPredictionTimestampUtc,
          })
      : [];
  const predictionsBySegmentId = new Map(
    predictions
      .filter((prediction): prediction is NonNullable<typeof prediction> =>
        Boolean(prediction),
      )
      .map((prediction) => [prediction.segmentId, prediction]),
  );
  const predictedSegments = predictions.filter(Boolean).length;
  const predictionRows = predictions.filter(
    (prediction): prediction is NonNullable<typeof prediction> => Boolean(prediction),
  );
  const observationRows = latestObservations.filter(
    (observation): observation is NonNullable<typeof observation> => Boolean(observation),
  );
  const liveWindow = getLiveWindowPayload();

  return {
    corridor: buildCorridorPayload(segments.length),
    generatedAtUtc: new Date().toISOString(),
    liveWindow,
    model: {
      version: modelRun?.version ?? null,
      runId: modelRun?.runId ?? null,
      modelName: modelRun?.modelName ?? null,
      artifactPath: modelRun?.artifactPath ?? null,
      createdAt: toIsoString(modelRun?.createdAt),
      warnings: parseModelWarnings(modelRun?.metricsJson),
    },
    freshness: {
      status: getPredictionFreshness(predictions, latestPredictionTimestampUtc, liveWindow),
      latestPredictionTimestampUtc: toIsoString(latestPredictionTimestampUtc),
      predictedSegments,
      missingSegments: segments.length - predictedSegments,
    },
    summary: {
      predictionCounts: countLabels(predictionRows),
      averageConfidence: average(predictionRows.map((prediction) => prediction.confidence)),
      lowConfidenceCount: predictionRows.filter(
        (prediction) => prediction.confidence !== null && prediction.confidence < 0.55,
      ).length,
      currentCongestionCounts: countLabels(observationRows),
    },
    segments: segments.map((segment) => {
      const prediction = predictionsBySegmentId.get(segment.segmentId);
      const observation = observationsBySegmentId.get(segment.segmentId);

      return {
        segmentId: segment.segmentId,
        roadName: segment.roadName,
        roadType: segment.roadType,
        latitude: segment.latitude,
        longitude: segment.longitude,
        order: segment.sortOrder,
        prediction: prediction ? mapPrediction(prediction) : null,
        latestObservation: observation ? mapObservation(observation) : null,
      };
    }),
  };
}

export async function getPredictionTrendPayload() {
  await syncSegmentsFromDefinition();

  const segments = await listSegments();
  const segmentIds = segments.map((segment) => segment.segmentId);
  const latestModelRun = await getLatestModelRun();
  const latestObservations = await getLatestTrafficObservations(segmentIds);
  const snapshot = await resolvePredictionSnapshot({
    segmentIds,
    preferredModelVersion: latestModelRun?.version ?? null,
  });
  const modelRun = snapshot
    ? snapshot.modelVersion === latestModelRun?.version
      ? latestModelRun
      : await getModelRunByVersion(snapshot.modelVersion)
    : latestModelRun;
  const latestPredictionTimestampUtc = snapshot?.timestampUtc ?? null;
  const predictionFeatureTimestampUtc = latestPredictionTimestampUtc
    ? getFeatureTimestampForPrediction(latestPredictionTimestampUtc)
    : null;
  const predictions =
    snapshot && latestPredictionTimestampUtc
      ? snapshot.createdAt
        ? await getPredictionsForSegmentsInCreatedBatch({
            segmentIds,
            modelVersion: snapshot.modelVersion,
            createdAt: snapshot.createdAt,
          })
        : await getPredictionsForSegmentsAtTimestamp({
            segmentIds,
            modelVersion: snapshot.modelVersion,
            timestampUtc: latestPredictionTimestampUtc,
          })
      : [];
  const latestFeatures = predictionFeatureTimestampUtc
    ? await getLatestFeatureSnapshotsForSegments({
        segmentIds,
        featureVersion: FEATURE_VERSION,
        atOrBeforeUtc: predictionFeatureTimestampUtc,
      })
    : await getLatestFeatureSnapshotsForSegments({
        segmentIds,
        featureVersion: FEATURE_VERSION,
      });
  const recentFeatureGroups = await getRecentFeatureSnapshotsForSegments({
    segmentIds,
    featureVersion: FEATURE_VERSION,
    takePerSegment: 4,
    atOrBeforeUtc: predictionFeatureTimestampUtc ?? undefined,
  });
  const predictionsBySegmentId = new Map(
    predictions
      .filter((prediction): prediction is NonNullable<typeof prediction> =>
        Boolean(prediction),
      )
      .map((prediction) => [prediction.segmentId, prediction]),
  );
  const featuresBySegmentId = new Map(
    latestFeatures
      .filter((feature): feature is NonNullable<typeof feature> => Boolean(feature))
      .map((feature) => [feature.segmentId, feature]),
  );
  const observationsBySegmentId = new Map(
    latestObservations
      .filter((observation): observation is NonNullable<typeof observation> =>
        Boolean(observation),
      )
      .map((observation) => [observation.segmentId, observation]),
  );
  const recentFeaturesBySegmentId = new Map(
    recentFeatureGroups.map((features, index) => [segmentIds[index], features]),
  );
  const trendCounts = {
    improving: 0,
    stable: 0,
    worsening: 0,
    uncertain: 0,
  };

  const trendSegments = segments.map((segment) => {
    const prediction = predictionsBySegmentId.get(segment.segmentId) ?? null;
    const feature = featuresBySegmentId.get(segment.segmentId) ?? null;
    const observation = observationsBySegmentId.get(segment.segmentId) ?? null;
    const trend = getTrend({ feature, prediction, observation });
    const recentFeatures = recentFeaturesBySegmentId.get(segment.segmentId) ?? [];
    trendCounts[trend.trend] += 1;

    return {
      segmentId: segment.segmentId,
      roadName: segment.roadName,
      order: segment.sortOrder,
      latestFeatureTimestampUtc: toIsoString(feature?.timestampUtc),
      latestObservedLabel: observation?.congestionLabel ?? null,
      predictedLabel: prediction?.predictedLabel ?? null,
      confidence: prediction?.confidence ?? null,
      speedChangeRate: feature?.speedChangeRate ?? null,
      recentAverageSpeed: average(
        recentFeatures.map((recentFeature) => recentFeature.recentObservedSpeed),
      ),
      trend: trend.trend,
      reason: trend.reason,
    };
  });
  const predictionRows = predictions.filter(
    (prediction): prediction is NonNullable<typeof prediction> => Boolean(prediction),
  );

  return {
    corridor: buildCorridorPayload(segments.length),
    generatedAtUtc: new Date().toISOString(),
    modelVersion: modelRun?.version ?? null,
    summary: {
      ...trendCounts,
      averageConfidence: average(predictionRows.map((prediction) => prediction.confidence)),
    },
    segments: trendSegments,
  };
}
