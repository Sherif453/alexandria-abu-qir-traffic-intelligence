import { Prisma } from "@prisma/client";

import { prisma } from "@/lib/db";

export async function getLatestFeatureSnapshotsForSegments(params: {
  segmentIds: string[];
  featureVersion?: string;
  atOrBeforeUtc?: Date;
}) {
  if (params.segmentIds.length === 0) {
    return [];
  }

  return prisma.$transaction(
    params.segmentIds.map((segmentId) =>
      prisma.featureSnapshot.findFirst({
        where: {
          segmentId,
          featureVersion: params.featureVersion,
          ...(params.atOrBeforeUtc
            ? {
                timestampUtc: {
                  lte: params.atOrBeforeUtc,
                },
              }
            : {}),
        },
        orderBy: {
          timestampUtc: "desc",
        },
      }),
    ),
  );
}

export async function getRecentFeatureSnapshotsForSegments(params: {
  segmentIds: string[];
  featureVersion?: string;
  takePerSegment: number;
  atOrBeforeUtc?: Date;
}) {
  if (params.segmentIds.length === 0) {
    return [];
  }

  return prisma.$transaction(
    params.segmentIds.map((segmentId) =>
      prisma.featureSnapshot.findMany({
        where: {
          segmentId,
          featureVersion: params.featureVersion,
          ...(params.atOrBeforeUtc
            ? {
                timestampUtc: {
                  lte: params.atOrBeforeUtc,
                },
              }
            : {}),
        },
        orderBy: {
          timestampUtc: "desc",
        },
        take: params.takePerSegment,
      }),
    ),
  );
}

export async function getFeatureSnapshotsForSegmentsAtTimestamp(params: {
  segmentIds: string[];
  featureVersion?: string;
  timestampUtc: Date;
}) {
  if (params.segmentIds.length === 0) {
    return [];
  }

  const features = params.featureVersion
    ? await prisma.$queryRaw<
        Awaited<ReturnType<typeof prisma.featureSnapshot.findMany>>
      >`
        SELECT *
        FROM "FeatureSnapshot"
        WHERE "featureVersion" = ${params.featureVersion}
          AND "timestampUtc" = ${params.timestampUtc.toISOString()}
          AND "segmentId" IN (${Prisma.join(params.segmentIds)})
        ORDER BY "segmentId" ASC, "createdAt" DESC
      `
    : await prisma.$queryRaw<
        Awaited<ReturnType<typeof prisma.featureSnapshot.findMany>>
      >`
        SELECT *
        FROM "FeatureSnapshot"
        WHERE "timestampUtc" = ${params.timestampUtc.toISOString()}
          AND "segmentId" IN (${Prisma.join(params.segmentIds)})
        ORDER BY "segmentId" ASC, "createdAt" DESC
      `;

  if (features.length > 0) {
    const featuresBySegmentId = new Map<string, (typeof features)[number]>();

    for (const feature of features) {
      if (!featuresBySegmentId.has(feature.segmentId)) {
        featuresBySegmentId.set(feature.segmentId, feature);
      }
    }

    return params.segmentIds.map((segmentId) => featuresBySegmentId.get(segmentId) ?? null);
  }

  return prisma.$transaction(
    params.segmentIds.map((segmentId) =>
      prisma.featureSnapshot.findFirst({
        where: {
          segmentId,
          featureVersion: params.featureVersion,
          timestampUtc: params.timestampUtc,
        },
        orderBy: {
          createdAt: "desc",
        },
      }),
    ),
  );
}
