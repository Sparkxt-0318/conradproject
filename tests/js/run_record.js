// Hashes strings and builds a small log with the phone's code; tests/test_app.py checks both in Python.
const path = require("path");
const R = require(path.join(__dirname, "..", "..", "soilsight", "app", "static", "sha256.js"));
const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
const entries = [];
let prev = R.GENESIS;
input.events.forEach((e, i) => { const item = R.makeEntry(i + 1, e.time, e.type, e.data, prev); entries.push(item); prev = item.hash; });
process.stdout.write(JSON.stringify({ hashes: input.strings.map(R.sha256), entries }));
