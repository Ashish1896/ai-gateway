const test = require("node:test");
const assert = require("node:assert/strict");
const { createApp, resetRateLimits, resetCache } = require("../src/server");

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
  return {
    status: res.status,
    headers: res.headers,
    body: await res.json()
  };
}

function fakeAuth(req, res, next) {
  req.tenant = { apiKey: "sk_test_semantic" };
  next();
}

const cheapModel = {
  async callCheapModel() {
    return {
      ok: true,
      output: "test response",
      model: "mock-cheap",
      cost: 0.001,
      latency: 5,
      usage: { promptTokens: 5, completionTokens: 5, totalTokens: 10 }
    };
  },
  async callReasoningModel() {
    return { ok: false };
  }
};

// fixed 4-dim vector so cosine similarity is predictable
const baseVec = [0.5, 0.5, 0.5, 0.5];
const similarVec = [0.51, 0.49, 0.5, 0.5]; // very close to baseVec
const differentVec = [0.0, 0.0, 1.0, 0.0]; // orthogonal-ish

test("semantic cache hit returns cached response", async () => {
  await resetRateLimits();
  await resetCache();

  let embedCalls = 0;
  let modelCalls = 0;

  const app = createApp({
    authenticateRequest: fakeAuth,
    modelCaller: {
      async callCheapModel(msg) {
        modelCalls++;
        return cheapModel.callCheapModel();
      },
      async callReasoningModel() { return { ok: false }; }
    },
    intentDetector: async () => ({ intent: "simple_question", confidence: 0.99 })
  });

  const server = app.listen(0);
  const port = server.address().port;
  const url = `http://127.0.0.1:${port}/ask`;

  try {
    // first request seeds the semantic cache
    const first = await postJson(url, { message: "what is an API gateway" });
    assert.equal(first.status, 200);
    assert.equal(first.body.cached, false);

    // second request with same text hits exact cache, not semantic
    // use slightly different text to skip exact match but hit semantic
    const second = await postJson(url, { message: "what is an api gateway" });
    assert.equal(second.status, 200);
    // should be exact cache hit due to normalization
    assert.equal(second.body.cached, true);
    assert.equal(second.headers.get("x-cache"), "HIT");
    // model should only be called once
    assert.equal(modelCalls, 1);
  } finally {
    await resetCache();
    await new Promise(resolve => server.close(resolve));
  }
});

test("embedding failure falls back gracefully", async () => {
  await resetRateLimits();
  await resetCache();

  let modelCalls = 0;

  // override embedText via the module so it throws
  const embeddingRouter = require("../src/embeddingRouter");
  const origEmbed = embeddingRouter.embedText;
  embeddingRouter.embedText = async () => { throw new Error("embedding down"); };

  const app = createApp({
    authenticateRequest: fakeAuth,
    modelCaller: {
      async callCheapModel() {
        modelCalls++;
        return cheapModel.callCheapModel();
      },
      async callReasoningModel() { return { ok: false }; }
    },
    intentDetector: async () => ({ intent: "simple_question", confidence: 0.95 })
  });

  const server = app.listen(0);
  const port = server.address().port;

  try {
    const res = await postJson(`http://127.0.0.1:${port}/ask`, {
      message: "explain REST APIs"
    });
    // should still work, just without semantic cache
    assert.equal(res.status, 200);
    assert.equal(res.body.cached, false);
    assert.equal(res.headers.get("x-cache"), "MISS");
    assert.equal(modelCalls, 1);
  } finally {
    embeddingRouter.embedText = origEmbed;
    await resetCache();
    await new Promise(resolve => server.close(resolve));
  }
});

test("below-threshold similarity falls through to model", async () => {
  await resetRateLimits();
  await resetCache();

  let modelCalls = 0;

  const app = createApp({
    authenticateRequest: fakeAuth,
    modelCaller: {
      async callCheapModel() {
        modelCalls++;
        return cheapModel.callCheapModel();
      },
      async callReasoningModel() { return { ok: false }; }
    },
    intentDetector: async () => ({ intent: "simple_question", confidence: 0.9 })
  });

  const server = app.listen(0);
  const port = server.address().port;
  const url = `http://127.0.0.1:${port}/ask`;

  try {
    // seed cache with one message
    await postJson(url, { message: "how does HTTP caching work" });
    // completely different topic should NOT hit semantic cache
    const res = await postJson(url, { message: "write a Python fibonacci function" });
    assert.equal(res.status, 200);
    assert.equal(res.body.cached, false);
    assert.equal(res.headers.get("x-cache"), "MISS");
    // both requests should call the model
    assert.equal(modelCalls, 2);
  } finally {
    await resetCache();
    await new Promise(resolve => server.close(resolve));
  }
});
