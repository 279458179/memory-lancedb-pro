import { describe, it } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const {
  classifyReflectionRetry,
  computeReflectionRetryDelayMs,
  isReflectionNonRetryError,
  isTransientReflectionUpstreamError,
  runWithReflectionTransientRetryOnce,
} = jiti("../src/reflection-retry.ts");

describe("reflection transient retry classifier", () => {
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
