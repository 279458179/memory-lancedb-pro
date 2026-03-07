import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import jitiFactory from "jiti";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const pluginSdkStubPath = path.resolve(testDir, "helpers", "openclaw-plugin-sdk-stub.mjs");
const jiti = jitiFactory(import.meta.url, {
  interopDefault: true,
  alias: {
    "openclaw/plugin-sdk": pluginSdkStubPath,
  },
});

const { readSessionConversationWithResetFallback, parsePluginConfig } = jiti("../index.ts");
const { getDisplayCategoryTag } = jiti("../src/reflection-metadata.ts");
const {
  classifyReflectionRetry,
  computeReflectionRetryDelayMs,
  isReflectionNonRetryError,
  isTransientReflectionUpstreamError,
  runWithReflectionTransientRetryOnce,
} = jiti("../src/reflection-retry.ts");
const {
  storeReflectionToLanceDB,
  loadAgentReflectionSlicesFromEntries,
  REFLECTION_DERIVE_LOGISTIC_K,
  REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS,
  REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT,
} = jiti("../src/reflection-store.ts");

function messageLine(role, text, ts) {
  return JSON.stringify({
    type: "message",
    timestamp: ts,
    message: {
      role,
      content: [{ type: "text", text }],
    },
  });
}

function makeEntry({ timestamp, metadata, category = "reflection", scope = "global" }) {
  return {
    id: `mem-${Math.random().toString(36).slice(2, 8)}`,
    text: "reflection-entry",
    vector: [],
    category,
    scope,
    importance: 0.7,
    timestamp,
    metadata: JSON.stringify(metadata),
  };
}

function baseConfig() {
  return {
    embedding: {
      apiKey: "test-api-key",
    },
  };
}

describe("memory reflection", () => {
  describe("command:new/reset session fallback helper", () => {
    let workDir;

    beforeEach(() => {
      workDir = mkdtempSync(path.join(tmpdir(), "reflection-fallback-test-"));
    });

    afterEach(() => {
      rmSync(workDir, { recursive: true, force: true });
    });

    it("falls back to latest reset snapshot when current session has only slash/control messages", async () => {
      const sessionsDir = path.join(workDir, "sessions");
      const sessionPath = path.join(sessionsDir, "abc123.jsonl");
      const resetOldPath = `${sessionPath}.reset.1700000000`;
      const resetNewPath = `${sessionPath}.reset.1700000001`;
      mkdirSync(sessionsDir, { recursive: true });

      writeFileSync(
        sessionPath,
        [messageLine("user", "/new", 1), messageLine("assistant", "/note self-improvement (before reset): ...", 2)].join("\n") + "\n",
        "utf-8"
      );
      writeFileSync(
        resetOldPath,
        [messageLine("user", "old reset snapshot", 3), messageLine("assistant", "old reset reply", 4)].join("\n") + "\n",
        "utf-8"
      );
      writeFileSync(
        resetNewPath,
        [
          messageLine("user", "Please keep responses concise and factual.", 5),
          messageLine("assistant", "Acknowledged. I will keep responses concise and factual.", 6),
        ].join("\n") + "\n",
        "utf-8"
      );

      const oldTime = new Date("2024-01-01T00:00:00Z");
      const newTime = new Date("2024-01-01T00:00:10Z");
      utimesSync(resetOldPath, oldTime, oldTime);
      utimesSync(resetNewPath, newTime, newTime);

      const conversation = await readSessionConversationWithResetFallback(sessionPath, 10);
      assert.ok(conversation);
      assert.match(conversation, /user: Please keep responses concise and factual\./);
      assert.match(conversation, /assistant: Acknowledged\. I will keep responses concise and factual\./);
      assert.doesNotMatch(conversation, /old reset snapshot/);
      assert.doesNotMatch(conversation, /^user:\s*\/new/m);
    });
  });

  describe("display category tags", () => {
    it("uses scope tag for reflection entries", () => {
      assert.equal(
        getDisplayCategoryTag({
          category: "reflection",
          scope: "project-a",
          metadata: JSON.stringify({ type: "memory-reflection", invariants: ["Always verify output"] }),
        }),
        "reflection:project-a"
      );

      assert.equal(
        getDisplayCategoryTag({
          category: "reflection",
          scope: "project-b",
          metadata: JSON.stringify({
            type: "memory-reflection",
            reflectionVersion: 3,
            invariants: ["Always verify output"],
            derived: ["Next run keep prompts short."],
          }),
        }),
        "reflection:project-b"
      );
    });

    it("uses scope tag for reflection rows with optional metadata fields", () => {
      assert.equal(
        getDisplayCategoryTag({
          category: "reflection",
          scope: "global",
          metadata: JSON.stringify({
            type: "memory-reflection",
            reflectionVersion: 3,
            invariants: ["Always keep steps auditable."],
            derived: ["Next run keep verification concise."],
            deriveBaseWeight: 0.35,
          }),
        }),
        "reflection:global"
      );
    });

    it("preserves non-reflection display categories", () => {
      assert.equal(
        getDisplayCategoryTag({
          category: "fact",
          scope: "global",
          metadata: "{}",
        }),
        "fact:global"
      );
    });
  });

  describe("transient retry classifier", () => {
    it("classifies unexpected EOF as transient upstream error", () => {
      const isTransient = isTransientReflectionUpstreamError(new Error("unexpected EOF while reading upstream response"));
      assert.equal(isTransient, true);
    });

    it("classifies auth/billing/model/context/session/refusal errors as non-retry", () => {
      assert.equal(isReflectionNonRetryError(new Error("401 unauthorized: invalid api key")), true);
      assert.equal(isReflectionNonRetryError(new Error("insufficient credits for this request")), true);
      assert.equal(isReflectionNonRetryError(new Error("model not found: gpt-x")), true);
      assert.equal(isReflectionNonRetryError(new Error("context length exceeded")), true);
      assert.equal(isReflectionNonRetryError(new Error("session expired, please re-authenticate")), true);
      assert.equal(isReflectionNonRetryError(new Error("refusal due to safety policy")), true);
    });

    it("allows retry only in reflection scope with zero useful output and retryCount=0", () => {
      const allowed = classifyReflectionRetry({
        inReflectionScope: true,
        retryCount: 0,
        usefulOutputChars: 0,
        error: new Error("upstream temporarily unavailable (503)"),
      });
      assert.equal(allowed.retryable, true);
      assert.equal(allowed.reason, "transient_upstream_failure");

      const notScope = classifyReflectionRetry({
        inReflectionScope: false,
        retryCount: 0,
        usefulOutputChars: 0,
        error: new Error("unexpected EOF"),
      });
      assert.equal(notScope.retryable, false);
      assert.equal(notScope.reason, "not_reflection_scope");

      const hadOutput = classifyReflectionRetry({
        inReflectionScope: true,
        retryCount: 0,
        usefulOutputChars: 12,
        error: new Error("unexpected EOF"),
      });
      assert.equal(hadOutput.retryable, false);
      assert.equal(hadOutput.reason, "useful_output_present");

      const retryUsed = classifyReflectionRetry({
        inReflectionScope: true,
        retryCount: 1,
        usefulOutputChars: 0,
        error: new Error("unexpected EOF"),
      });
      assert.equal(retryUsed.retryable, false);
      assert.equal(retryUsed.reason, "retry_already_used");
    });

    it("computes jitter delay in the required 1-3s range", () => {
      assert.equal(computeReflectionRetryDelayMs(() => 0), 1000);
      assert.equal(computeReflectionRetryDelayMs(() => 0.5), 2000);
      assert.equal(computeReflectionRetryDelayMs(() => 1), 3000);
    });
  });

  describe("runWithReflectionTransientRetryOnce", () => {
    it("retries once and succeeds for transient failures", async () => {
      let attempts = 0;
      const sleeps = [];
      const logs = [];
      const retryState = { count: 0 };

      const result = await runWithReflectionTransientRetryOnce({
        scope: "reflection",
        runner: "embedded",
        retryState,
        execute: async () => {
          attempts += 1;
          if (attempts === 1) {
            throw new Error("unexpected EOF from provider");
          }
          return "ok";
        },
        random: () => 0,
        sleep: async (ms) => {
          sleeps.push(ms);
        },
        onLog: (level, message) => logs.push({ level, message }),
      });

      assert.equal(result, "ok");
      assert.equal(attempts, 2);
      assert.equal(retryState.count, 1);
      assert.deepEqual(sleeps, [1000]);
      assert.equal(logs.length, 2);
      assert.match(logs[0].message, /transient upstream failure detected/i);
      assert.match(logs[0].message, /retrying once in 1000ms/i);
      assert.match(logs[1].message, /retry succeeded/i);
    });

    it("does not retry non-transient failures", async () => {
      let attempts = 0;
      const retryState = { count: 0 };

      await assert.rejects(
        runWithReflectionTransientRetryOnce({
          scope: "reflection",
          runner: "cli",
          retryState,
          execute: async () => {
            attempts += 1;
            throw new Error("invalid api key");
          },
          sleep: async () => { },
        }),
        /invalid api key/i
      );

      assert.equal(attempts, 1);
      assert.equal(retryState.count, 0);
    });

    it("does not loop: exhausted after one retry", async () => {
      let attempts = 0;
      const logs = [];
      const retryState = { count: 0 };

      await assert.rejects(
        runWithReflectionTransientRetryOnce({
          scope: "distiller",
          runner: "cli",
          retryState,
          execute: async () => {
            attempts += 1;
            throw new Error("service unavailable 503");
          },
          random: () => 0.1,
          sleep: async () => { },
          onLog: (level, message) => logs.push({ level, message }),
        }),
        /service unavailable/i
      );

      assert.equal(attempts, 2);
      assert.equal(retryState.count, 1);
      assert.equal(logs.length, 2);
      assert.match(logs[1].message, /retry exhausted/i);
    });
  });

  describe("single-entry persistence", () => {
    it("stores one combined reflection entry with invariant+derived slices and derive decay metadata", async () => {
      const storedEntries = [];
      const vectorSearchCalls = [];

      const result = await storeReflectionToLanceDB({
        reflectionText: [
          "## Invariants",
          "- Always confirm assumptions before changing files.",
          "## Derived",
          "- Next run verify reflection persistence with targeted tests.",
        ].join("\n"),
        sessionKey: "agent:main:session:abc",
        sessionId: "abc",
        agentId: "main",
        command: "command:reset",
        scope: "global",
        toolErrorSignals: [{ signatureHash: "deadbeef" }],
        runAt: 1_700_000_000_000,
        usedFallback: false,
        embedPassage: async (text) => [text.length],
        vectorSearch: async (vector) => {
          vectorSearchCalls.push(vector);
          return [];
        },
        store: async (entry) => {
          storedEntries.push(entry);
          return { ...entry, id: `id-${storedEntries.length}`, timestamp: 1_700_000_000_000 };
        },
      });

      assert.equal(result.stored, true);
      assert.deepEqual(result.storedKinds, ["combined"]);
      assert.equal(storedEntries.length, 1);
      assert.equal(vectorSearchCalls.length, 1);

      const stored = storedEntries[0];
      const meta = JSON.parse(stored.metadata);

      assert.equal(stored.category, "reflection");
      assert.equal(meta.reflectionVersion, 3);
      assert.deepEqual(meta.invariants, ["Always confirm assumptions before changing files."]);
      assert.deepEqual(meta.derived, ["Next run verify reflection persistence with targeted tests."]);
      assert.equal(meta.decayModel, "logistic");
      assert.equal(meta.decayK, REFLECTION_DERIVE_LOGISTIC_K);
      assert.equal(meta.decayMidpointDays, REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS);
      assert.equal(meta.deriveBaseWeight, 1);
    });
  });

  describe("reflection slice loading", () => {
    it("loads combined entries from both older and current metadata layouts", () => {
      const now = Date.UTC(2026, 2, 7);

      const entries = [
        makeEntry({
          timestamp: now - 30 * 60 * 1000,
          metadata: {
            type: "memory-reflection",
            agentId: "main",
            invariants: ["Legacy invariant still applies."],
            derived: ["Legacy derived delta still applies."],
            storedAt: now - 30 * 60 * 1000,
          },
        }),
        makeEntry({
          timestamp: now - 25 * 60 * 1000,
          metadata: {
            type: "memory-reflection",
            agentId: "main",
            reflectionVersion: 3,
            invariants: ["Current invariant applies too."],
            derived: ["Current derived delta still applies."],
            storedAt: now - 25 * 60 * 1000,
            decayModel: "logistic",
            decayMidpointDays: REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS,
            decayK: REFLECTION_DERIVE_LOGISTIC_K,
          },
        }),
      ];

      const slices = loadAgentReflectionSlicesFromEntries({
        entries,
        agentId: "main",
        now,
        deriveMaxAgeMs: 7 * 24 * 60 * 60 * 1000,
      });

      assert.ok(slices.invariants.includes("Legacy invariant still applies."));
      assert.ok(slices.invariants.includes("Current invariant applies too."));

      assert.ok(slices.derived.includes("Legacy derived delta still applies."));
      assert.ok(slices.derived.includes("Current derived delta still applies."));
    });
  });

  describe("derive logistic scoring", () => {
    it("ranks recent derived guidance using logistic decay and fallback base-weight", () => {
      const now = Date.UTC(2026, 2, 7);
      const day = 24 * 60 * 60 * 1000;

      const entries = [
        makeEntry({
          timestamp: now - 1 * day,
          metadata: {
            type: "memory-reflection",
            agentId: "main",
            derived: ["Fresh normal derive"],
            storedAt: now - 1 * day,
            deriveBaseWeight: 1,
            usedFallback: false,
            reflectionVersion: 3,
          },
        }),
        makeEntry({
          timestamp: now - 1 * day,
          metadata: {
            type: "memory-reflection",
            agentId: "main",
            derived: ["Fresh fallback derive"],
            storedAt: now - 1 * day,
            usedFallback: true,
            reflectionVersion: 3,
          },
        }),
        makeEntry({
          timestamp: now - 5 * day,
          metadata: {
            type: "memory-reflection",
            agentId: "main",
            derived: ["Older normal derive"],
            storedAt: now - 5 * day,
            usedFallback: false,
            reflectionVersion: 3,
          },
        }),
        makeEntry({
          timestamp: now - 2 * day,
          metadata: {
            type: "memory-reflection",
            agentId: "main",
            derived: ["Second recent derive signal"],
            storedAt: now - 2 * day,
            usedFallback: false,
          },
        }),
      ];

      const slices = loadAgentReflectionSlicesFromEntries({
        entries,
        agentId: "main",
        now,
        deriveMaxAgeMs: 10 * day,
      });

      assert.equal(slices.derived[0], "Fresh normal derive");
      assert.ok(slices.derived.includes("Second recent derive signal"));

      const fallbackIdx = slices.derived.indexOf("Fresh fallback derive");
      const olderIdx = slices.derived.indexOf("Older normal derive");
      assert.notEqual(fallbackIdx, -1);
      assert.notEqual(olderIdx, -1);
      assert.ok(fallbackIdx < olderIdx, "fallback should still rank above much older derive due recency, but below normal fresh derive");

      assert.equal(REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT, 0.35);
    });
  });

  describe("sessionStrategy legacy compatibility mapping", () => {
    it("maps legacy sessionMemory.enabled=true to systemSessionMemory", () => {
      const parsed = parsePluginConfig({
        ...baseConfig(),
        sessionMemory: { enabled: true },
      });
      assert.equal(parsed.sessionStrategy, "systemSessionMemory");
    });

    it("maps legacy sessionMemory.enabled=false to none", () => {
      const parsed = parsePluginConfig({
        ...baseConfig(),
        sessionMemory: { enabled: false },
      });
      assert.equal(parsed.sessionStrategy, "none");
    });

    it("prefers explicit sessionStrategy over legacy sessionMemory.enabled", () => {
      const parsed = parsePluginConfig({
        ...baseConfig(),
        sessionStrategy: "memoryReflection",
        sessionMemory: { enabled: false },
      });
      assert.equal(parsed.sessionStrategy, "memoryReflection");
    });

    it("defaults to systemSessionMemory when neither field is set", () => {
      const parsed = parsePluginConfig(baseConfig());
      assert.equal(parsed.sessionStrategy, "systemSessionMemory");
    });
  });
});
