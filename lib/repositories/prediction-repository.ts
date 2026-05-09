import { Prisma } from "@prisma/client";

import { prisma } from "@/lib/db";

export type PredictionSnapshotCandidate = {
  modelVersion: string;
  timestampUtc: Date;
  predictedSegments: number;
};

export async function getLatestPredictionTimestamp(params: { modelVersion: string }) {
  const prediction = await prisma.prediction.findFirst({
    where: {
      modelVersion: params.modelVersion,
    },
    orderBy: [
      {
        timestampUtc: "desc",
      },
      {
        createdAt: "desc",
      },
    ],
    select: {
      timestampUtc: true,
    },
  });

  return prediction?.timestampUtc ?? null;
}

export async function getPredictionsForSegmentsAtTimestamp(params: {
  segmentIds: string[];
  modelVersion: string;
  timestampUtc: Date;
}) {
  if (params.segmentIds.length === 0) {
    return [];
  }

  const predictions = await prisma.$queryRaw<
    Awaited<ReturnType<typeof prisma.prediction.findMany>>
  >`
    SELECT *
    FROM "Prediction"
    WHERE "modelVersion" = ${params.modelVersion}
      AND "timestampUtc" = ${params.timestampUtc.toISOString()}
      AND "segmentId" IN (${Prisma.join(params.segmentIds)})
    ORDER BY "segmentId" ASC, "createdAt" DESC
  `;

  const resolvedPredictions =
    predictions.length > 0
      ? predictions
      : await prisma.prediction.findMany({
          where: {
            segmentId: {
              in: params.segmentIds,
            },
            modelVersion: params.modelVersion,
            timestampUtc: params.timestampUtc,
          },
          orderBy: [
            {
              segmentId: "asc",
            },
            {
              createdAt: "desc",
            },
          ],
        });

  const predictionsBySegmentId = new Map<string, (typeof resolvedPredictions)[number]>();

  for (const prediction of resolvedPredictions) {
    if (!predictionsBySegmentId.has(prediction.segmentId)) {
      predictionsBySegmentId.set(prediction.segmentId, prediction);
    }
  }

  return params.segmentIds.map((segmentId) => predictionsBySegmentId.get(segmentId) ?? null);
}

export async function listPredictionsInRange(params: {
  segmentIds: string[];
  modelVersion: string;
  fromUtc: Date;
  toUtc: Date;
}) {
  if (params.segmentIds.length === 0) {
    return [];
  }

  return prisma.prediction.findMany({
    where: {
      segmentId: {
        in: params.segmentIds,
      },
      modelVersion: params.modelVersion,
      timestampUtc: {
        gte: params.fromUtc,
        lte: params.toUtc,
      },
    },
    orderBy: [
      {
        timestampUtc: "asc",
      },
      {
        segmentId: "asc",
      },
    ],
  });
}

export async function listPredictionSnapshotCandidates(params: {
  segmentIds: string[];
  modelVersion?: string;
  take?: number;
}): Promise<PredictionSnapshotCandidate[]> {
  if (params.segmentIds.length === 0) {
    return [];
  }

  const grouped = await prisma.prediction.groupBy({
    by: ["modelVersion", "timestampUtc"],
    where: {
      segmentId: {
        in: params.segmentIds,
      },
      ...(params.modelVersion ? { modelVersion: params.modelVersion } : {}),
    },
    _count: {
      segmentId: true,
    },
    orderBy: {
      timestampUtc: "desc",
    },
    take: params.take ?? 12,
  });

  return grouped.map((row) => ({
    modelVersion: row.modelVersion,
    timestampUtc: row.timestampUtc,
    predictedSegments: row._count.segmentId,
  }));
}
