const config = require("../config");

let extractorPromise = null;

async function loadFeatureExtractor() {
  const { env, pipeline } = await import("@huggingface/transformers");
  env.allowLocalModels = true;
  env.allowRemoteModels = true;

  return pipeline("feature-extraction", config.embedding.localModel);
}

function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = loadFeatureExtractor().catch(error => {
      extractorPromise = null;
      throw error;
    });
  }

  return extractorPromise;
}

function prepareInput(text) {
  const normalized = String(text || "").trim();
  if (!normalized) {
    return normalized;
  }

  const prefix = String(config.embedding.queryPrefix || "");
  return prefix ? `${prefix}${normalized}` : normalized;
}

function toVectors(output) {
  const list = typeof output?.tolist === "function" ? output.tolist() : [];
  if (!Array.isArray(list)) {
    return [];
  }

  if (Array.isArray(list[0])) {
    return list;
  }

  return [list];
}

async function embedWithBge(text) {
  const extractor = await getExtractor();
  const output = await extractor(prepareInput(text), {
    pooling: "mean",
    normalize: true
  });
  const [vector] = toVectors(output);

  if (!Array.isArray(vector) || vector.length === 0) {
    throw new Error(`BGE embedding model returned no vector for ${config.embedding.localModel}`);
  }

  return vector;
}

module.exports = {
  embedWithBge
};
