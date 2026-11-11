// Runs the phone's Brain on a scenario from a JSON file and prints its answers as JSON.
// Used by tests/test_app.py to compare the JavaScript Brain with the Python one.
const fs = require("fs");
const path = require("path");
const SoilBrain = require(path.join(__dirname, "..", "..", "soilsight", "app", "static", "brain.js"));
const scenario = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const brain = SoilBrain.create(scenario.package);
const out = { ndtr: (scenario.ndtr || []).map(SoilBrain.ndtr), steps: [] };
for (const tests of scenario.steps) {
  const t0 = Date.now();
  const post = brain.fit(tests);
  const map = brain.predict(post);
  const next = brain.nextSpot(post);
  out.steps.push({
    weights: post.comps.map((c) => c.weight),
    mean: Array.from(map.mean), sd: Array.from(map.sd), p: Array.from(map.p),
    zones: brain.zones(post), next, ms: Date.now() - t0,
  });
}
process.stdout.write(JSON.stringify(out));
