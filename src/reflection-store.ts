import type { MemoryEntry, MemorySearchResult } from "./store.js";
import { extractReflectionSlices, sanitizeReflectionSliceLines, type ReflectionSlices } from "./reflection-slices.js";
import { parseReflectionMetadata } from "./reflection-metadata.js";

export const REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS = 3;
export const REFLECTION_DERIVE_LOGISTIC_K = 1.2;
export const REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT = 0.35;
export const DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000;

type ReflectionStoreKind = "combined";

type ReflectionErrorSignalLike = {
  signatureHash: string;
};

interface ReflectionStorePayload {
  text: string;
  metadata: Record<string, unknown>;
  kind: ReflectionStoreKind;
}

interface BuildReflectionStorePayloadsParams {
  reflectionText: string;
  sessionKey: string;
  sessionId: string;
  agentId: string;
  command: string;
  scope: string;
  toolErrorSignals: ReflectionErrorSignalLike[];
  runAt: number;
  usedFallback: boolean;
}

export function buildReflectionStorePayloads(params: BuildReflectionStorePayloadsParams): {
  slices: ReflectionSlices;
  payloads: ReflectionStorePayload[];
} {
  const slices = extractReflectionSlices(params.reflectionText);
  if (slices.invariants.length === 0 && slices.derived.length === 0) {
    return { slices, payloads: [] };
  }

  const dateYmd = new Date(params.runAt).toISOString().split("T")[0];
  const deriveQuality = computeDerivedLineQuality(slices.derived.length);
  const deriveBaseWeight = params.usedFallback ? REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT : 1;
  const payloads: ReflectionStorePayload[] = [{
    kind: "combined",
    text: [
      `reflection · ${params.scope} · ${dateYmd}`,
      `Session Reflection (${new Date(params.runAt).toISOString()})`,
      `Session Key: ${params.sessionKey}`,
      `Session ID: ${params.sessionId}`,
      "",
      "Invariants:",
      ...(slices.invariants.length > 0 ? slices.invariants.map((x) => `- ${x}`) : ["- (none captured)"]),
      "",
      "Derived:",
      ...(slices.derived.length > 0 ? slices.derived.map((x) => `- ${x}`) : ["- (none captured)"]),
    ].join("\n"),
    metadata: {
      type: "memory-reflection",
      stage: "reflect-store",
      reflectionVersion: 3,
      sessionKey: params.sessionKey,
      sessionId: params.sessionId,
      agentId: params.agentId,
      command: params.command,
      storedAt: params.runAt,
      invariants: slices.invariants,
      derived: slices.derived,
      usedFallback: params.usedFallback,
      errorSignals: params.toolErrorSignals.map((s) => s.signatureHash),
      decayModel: "logistic",
      decayMidpointDays: REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS,
      decayK: REFLECTION_DERIVE_LOGISTIC_K,
      deriveBaseWeight,
      deriveQuality,
      deriveSource: params.usedFallback ? "fallback" : "normal",
    },
  }];

  return { slices, payloads };
}

interface ReflectionStoreDeps {
  embedPassage: (text: string) => Promise<number[]>;
  vectorSearch: (
    vector: number[],
    limit?: number,
    minScore?: number,
    scopeFilter?: string[]
  ) => Promise<MemorySearchResult[]>;
  store: (entry: Omit<MemoryEntry, "id" | "timestamp">) => Promise<MemoryEntry>;
}

interface StoreReflectionToLanceDBParams extends BuildReflectionStorePayloadsParams, ReflectionStoreDeps {
  dedupeThreshold?: number;
}

export async function storeReflectionToLanceDB(params: StoreReflectionToLanceDBParams): Promise<{
  stored: boolean;
  slices: ReflectionSlices;
  storedKinds: ReflectionStoreKind[];
}> {
  const { slices, payloads } = buildReflectionStorePayloads(params);
  const storedKinds: ReflectionStoreKind[] = [];
  const dedupeThreshold = Number.isFinite(params.dedupeThreshold) ? Number(params.dedupeThreshold) : 0.97;

  for (const payload of payloads) {
    const vector = await params.embedPassage(payload.text);
    const existing = await params.vectorSearch(vector, 1, 0.1, [params.scope]);
    if (existing.length > 0 && existing[0].score > dedupeThreshold) {
      continue;
    }

    await params.store({
      text: payload.text,
      vector,
      category: "reflection",
      scope: params.scope,
      importance: 0.75,
      metadata: JSON.stringify(payload.metadata),
    });
    storedKinds.push(payload.kind);
  }

  return { stored: storedKinds.length > 0, slices, storedKinds };
}

export interface LoadReflectionSlicesParams {
  entries: MemoryEntry[];
  agentId: string;
  now?: number;
  deriveMaxAgeMs?: number;
}

export function loadAgentReflectionSlicesFromEntries(params: LoadReflectionSlicesParams): {
  invariants: string[];
  derived: string[];
} {
  const now = Number.isFinite(params.now) ? Number(params.now) : Date.now();
  const deriveMaxAgeMs = Number.isFinite(params.deriveMaxAgeMs)
    ? Math.max(0, Number(params.deriveMaxAgeMs))
    : DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS;

  const reflections = params.entries
    .map((entry) => ({ entry, metadata: parseReflectionMetadata(entry.metadata) }))
    .filter(({ metadata }) => {
      if (metadata.type !== "memory-reflection") return false;
      const owner = typeof metadata.agentId === "string" ? metadata.agentId.trim() : "";
      if (!owner) return true;
      return owner === params.agentId || owner === "main";
    })
    .sort((a, b) => b.entry.timestamp - a.entry.timestamp)
    .slice(0, 40);

  const invariants: string[] = [];
  for (const { metadata } of reflections) {
    const inv = sanitizeReflectionSliceLines(toStringArray(metadata.invariants));
    for (const item of inv) {
      if (!item || invariants.includes(item)) continue;
      invariants.push(item);
      if (invariants.length >= 8) break;
    }
    if (invariants.length >= 8) break;
  }

  type WeightedLine = { line: string; score: number; latestTs: number };
  const lineScores = new Map<string, WeightedLine>();

  for (const { entry, metadata } of reflections) {
    const derivedLines = sanitizeReflectionSliceLines(toStringArray(metadata.derived));
    if (derivedLines.length === 0) continue;

    const timestamp = metadataTimestamp(metadata, entry.timestamp);
    if (now - timestamp > deriveMaxAgeMs) continue;

    const ageDays = Math.max(0, (now - timestamp) / 86_400_000);
    const decayMidpointDays = readPositiveNumber(metadata.decayMidpointDays, REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS);
    const decayK = readPositiveNumber(metadata.decayK, REFLECTION_DERIVE_LOGISTIC_K);
    const logistic = 1 / (1 + Math.exp(decayK * (ageDays - decayMidpointDays)));
    const deriveBaseWeight = resolveDeriveBaseWeight(metadata);
    const deriveQuality = readClampedNumber(metadata.deriveQuality, computeDerivedLineQuality(derivedLines.length), 0.2, 1);
    const entryWeight = logistic * deriveBaseWeight * deriveQuality;
    if (entryWeight <= 0) continue;

    for (const line of derivedLines) {
      const current = lineScores.get(line);
      if (!current) {
        lineScores.set(line, { line, score: entryWeight, latestTs: timestamp });
        continue;
      }
      current.score += entryWeight;
      if (timestamp > current.latestTs) current.latestTs = timestamp;
    }
  }

  const derived = [...lineScores.values()]
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      if (b.latestTs !== a.latestTs) return b.latestTs - a.latestTs;
      return a.line.localeCompare(b.line);
    })
    .slice(0, 10)
    .map((item) => item.line);

  return { invariants, derived };
}

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => String(item).trim())
    .filter(Boolean);
}

function metadataTimestamp(metadata: Record<string, unknown>, fallbackTs: number): number {
  const storedAt = Number(metadata.storedAt);
  if (Number.isFinite(storedAt) && storedAt > 0) return storedAt;
  return Number.isFinite(fallbackTs) ? fallbackTs : Date.now();
}

function readPositiveNumber(value: unknown, fallback: number): number {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return fallback;
  return num;
}

function readClampedNumber(value: unknown, fallback: number, min: number, max: number): number {
  const num = Number(value);
  const resolved = Number.isFinite(num) ? num : fallback;
  return Math.max(min, Math.min(max, resolved));
}

export function computeDerivedLineQuality(nonPlaceholderLineCount: number): number {
  const n = Number.isFinite(nonPlaceholderLineCount) ? Math.max(0, Math.floor(nonPlaceholderLineCount)) : 0;
  if (n <= 0) return 0.2;
  // Small quality factor boost for richer non-placeholder derive outputs.
  return Math.min(1, 0.55 + Math.min(6, n) * 0.075);
}

function resolveDeriveBaseWeight(metadata: Record<string, unknown>): number {
  const explicit = Number(metadata.deriveBaseWeight);
  if (Number.isFinite(explicit) && explicit > 0) {
    return Math.max(0.1, Math.min(1, explicit));
  }
  if (metadata.usedFallback === true) {
    return REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT;
  }
  return 1;
}
