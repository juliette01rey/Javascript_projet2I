import { readFileSync, writeFileSync, existsSync } from "fs";

const DICT_SIZE = 20_000;
const MODEL_PATH = "./data/markov-model.json";
const CORPUS_PATH = "./data/corpus.txt";

const FUNCTION_WORDS = new Set([
  "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
  "lui", "eux",
  "le", "la", "les", 
  "un", "une", "des", 
  "du", "de", "que",
  "et", "est", "en",
  "qui", 
  "se", "sa", "son", "ses", "leur", "leurs", "mon", "ton", "ma", "ta", "nos", "vos",
  "ce", "cet", "cette", "ces", "y", "à",
  "au", "aux", "par", "pour", "sur", "sous", "dans", "avec", "sans", "mais",
  "ou", "donc", "or", "ni", "car", "ne", "pas", "plus", "très", "bien",
  "tout", "tous", "toute", "toutes", 
  "même", "autre", "autres", 
]);

export function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-zàâäéèêëïîôùûüçœæ'\s-]/g, " ")
    .split(/\s+/)
    .map((w) => w.replace(/^['-]+|['-]+$/g, ""))
    .filter((w) => w.length > 1);
}

export function buildDictionary(tokens, maxSize = DICT_SIZE) {
  const freq = {};
  for (const token of tokens) {
    freq[token] = (freq[token] ?? 0) + 1;
  }
  const total = tokens.length;
  return Object.fromEntries(
    Object.entries(freq)
      .sort(([, a], [, b]) => b - a)
      .slice(0, maxSize)
      .map(([word, count]) => [word, count / total])
  );
}

export function computeWeights(freqMap) {
  return Object.fromEntries(
    Object.entries(freqMap).map(([word, normalizedFreq]) => {
      const penalty = FUNCTION_WORDS.has(word) ? 0.15 : 1.0;
      return [word, normalizedFreq * penalty];
    })
  );
}

export function train(tokens) {
  const transitions = new Map();
  for (let i = 0; i < tokens.length - 1; i++) {
    const current = tokens[i];
    const next = tokens[i + 1];
    if (!transitions.has(current)) {
      transitions.set(current, new Map());
    }
    const nexts = transitions.get(current);
    nexts.set(next, (nexts.get(next) ?? 0) + 1);
  }
  return transitions;
}

export function predict(transitions, weights, word, n = 5) {
  const nexts = transitions.get(word.toLowerCase());
  if (!nexts) return [];
  return [...nexts.entries()]
    .map(([w, count]) => ({
      word: w,
      score: count * (weights[w] ?? 0.001),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, n)
    .map((entry) => entry.word);
}

export function saveModel(transitions, weights, path = MODEL_PATH) {
  const serializable = {
    weights,
    transitions: Object.fromEntries(
      [...transitions.entries()].map(([k, v]) => [k, Object.fromEntries(v)])
    ),
  };
  writeFileSync(path, JSON.stringify(serializable));
  console.log(`✓ Modèle sauvegardé → ${path}`);
}

export function loadModel(path = MODEL_PATH) {
  const raw = JSON.parse(readFileSync(path, "utf8"));
  const transitions = new Map(
    Object.entries(raw.transitions).map(([k, v]) => [
      k,
      new Map(Object.entries(v)),
    ])
  );
  return { transitions, weights: raw.weights };
}

export function createPredictor(modelPath = MODEL_PATH) {
  const { transitions, weights } = loadModel(modelPath);
  return {
    top5: (word) => predict(transitions, weights, word),
  };
}

const isTrainMode = process.argv.includes("--train");

if (isTrainMode) {
  console.log("Lecture du corpus...");
  const text = readFileSync(CORPUS_PATH, "utf8");
  console.log("Tokenisation...");
  const tokens = tokenize(text);
  console.log(`Construction du dictionnaire (${DICT_SIZE} mots max)...`);
  const dict = buildDictionary(tokens, DICT_SIZE);
  const dictTokens = tokens.filter((t) => dict[t] !== undefined);
  const weights = computeWeights(dict);
  console.log("Entraînement de la chaîne de Markov...");
  const transitions = train(dictTokens);
  writeFileSync("./data/dictionary.json", JSON.stringify(dict, null, 2));
  saveModel(transitions, weights);
  console.log(`✓ Dictionnaire : ${Object.keys(dict).length} mots`);
  console.log(`✓ Transitions  : ${transitions.size} entrées`);
} else {
  if (!existsSync(MODEL_PATH)) {
    console.error("Aucun modèle trouvé. Lance d'abord : npm run train");
    process.exit(1);
  }
  const predictor = createPredictor();
  const testWords = ["bonjour", "maison", "et", "voiture", "le"];
  console.log("Prédictions top 5");
  for (const word of testWords) {
    const suggestions = predictor.top5(word);
    console.log(`"${word}" → [${suggestions.join(", ")}]`);
  }
}
