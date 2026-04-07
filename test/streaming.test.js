const test = require("node:test");
const assert = require("node:assert/strict");
const http = require("node:http");
const { Readable } = require("node:stream");
const { createApp, resetRateLimits, resetCache } = require("../src/server");
const { redisClient, connectRedis } = require("../src/redisClient");

const TENANT_KEY = "sk_test_stream";

function listen(app) {
  return new Promise(resolve => {
    const srv = app.listen(0, "127.0.0.1", () => resolve(srv));
  });
}

async function seedTenant(apiKey) {
  const tenant = {
    apiKey,
    tenantId: "tenant_test_stream",
    name: "test-stream",
    createdAt: Date.now(),
    requestsToday: 0,
    totalTokens: 0,
    totalCost: 0,
    lastReset: Date.now(),
    maxRequestsPerDay: 1000,
    maxTokensPerDay: 1_000_000,
    maxCostPerDay: 5
  };
  await redisClient.set(`tenant:${apiKey}`, JSON.stringify(tenant));
}

function fakeGroqStream(chunks) {
  const readable = new Readable({ read() {} });
  for (const text of chunks) {
    const payload = JSON.stringify({
      choices: [{ delta: { content: text } }]
    });
    readable.push(`data: ${payload}\n\n`);
  }
  readable.push("data: [DONE]\n\n");
  readable.push(null);
  return readable;
}

function requestSSE(server, body, headers = {}) {
  return new Promise((resolve, reject) => {
    const addr = server.address();
    const opts = {
      hostname: "127.0.0.1",
      port: addr.port,
      path: "/ask",
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "text/event-stream",
        authorization: `Bearer ${TENANT_KEY}`,
        ...headers
      }
    };

    const req = http.request(opts, res => {
      let raw = "";
      res.on("data", chunk => { raw += chunk; });
      res.on("end", () => {
        const events = raw
          .split("\n\n")
          .filter(b => b.trim())
          .map(block => {
            const lines = block.split("\n");
            const ev = lines.find(l => l.startsWith("event:"))?.slice(7) || "";
            const data = lines.find(l => l.startsWith("data:"))?.slice(5) || "{}";
            return { event: ev.trim(), data: JSON.parse(data) };
          });
        resolve({ status: res.statusCode, headers: res.headers, events });
      });
    });

    req.on("error", reject);
    req.write(JSON.stringify(body));
    req.end();
  });
}

test("SSE streaming", async (t) => {
  await connectRedis();
  await resetRateLimits();
  await resetCache();
  await seedTenant(TENANT_KEY);

  await t.test("streams tokens via SSE", async () => {
    const app = createApp({
      modelCaller: {
        callCheapModel: async () => ({
          ok: true, output: "test", model: "stub", cost: 0, latency: 1, usage: {}
        }),
        callReasoningModel: async () => ({
          ok: true, output: "test", model: "stub", cost: 0, latency: 1, usage: {}
        }),
        streamCheapModel: async () => ({
          stream: fakeGroqStream(["Hello", " world", "!"]),
          model: "test-model",
          startTime: Date.now()
        }),
        streamReasoningModel: async () => ({
          stream: fakeGroqStream(["Deep", " analysis"]),
          model: "test-reasoning",
          startTime: Date.now()
        })
      },
      intentDetector: async () => ({ intent: "general", confidence: 0.9, uncertain: false }),
      embedText: async () => [0, 0, 0]
    });

    const server = await listen(app);
    try {
      const { status, headers, events } = await requestSSE(server, { message: "hi" });
      assert.equal(status, 200);
      assert.equal(headers["content-type"], "text/event-stream");

      const route = events.find(e => e.event === "route");
      assert.ok(route, "should have route event");
      assert.ok(route.data.requestId);
      assert.ok(route.data.intent);

      const model = events.find(e => e.event === "model");
      assert.ok(model, "should have model event");
      assert.equal(model.data.model, "test-model");

      const tokens = events.filter(e => e.event === "token");
      assert.ok(tokens.length >= 3, "should have token events");
      assert.equal(tokens[0].data.token, "Hello");

      const done = events.find(e => e.event === "done");
      assert.ok(done, "should have done event");
      assert.equal(done.data.response, "Hello world!");
      assert.equal(done.data.model, "test-model");
    } finally {
      server.close();
    }
  });

  await t.test("falls back to JSON without Accept header", async () => {
    const app = createApp({
      modelCaller: {
        callCheapModel: async () => ({
          ok: true, output: "json response", model: "stub", cost: 0, latency: 1,
          usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 }
        }),
        callReasoningModel: async () => ({
          ok: true, output: "json response", model: "stub", cost: 0, latency: 1,
          usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 }
        })
      },
      intentDetector: async () => ({ intent: "general", confidence: 0.9, uncertain: false }),
      embedText: async () => [0, 0, 0]
    });

    const server = await listen(app);
    try {
      const addr = server.address();
      const res = await new Promise((resolve, reject) => {
        const req = http.request({
          hostname: "127.0.0.1",
          port: addr.port,
          path: "/ask",
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${TENANT_KEY}`
          }
        }, r => {
          let data = "";
          r.on("data", c => { data += c; });
          r.on("end", () => resolve({ status: r.statusCode, body: JSON.parse(data) }));
        });
        req.on("error", reject);
        req.write(JSON.stringify({ message: "hi" }));
        req.end();
      });

      assert.equal(res.status, 200);
      assert.equal(res.body.response, "json response");
      assert.equal(res.body.cached, false);
    } finally {
      server.close();
    }
  });

  await redisClient.del(`tenant:${TENANT_KEY}`);
});
