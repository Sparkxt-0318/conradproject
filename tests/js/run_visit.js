// A whole field visit on an emulated phone with no network, driven the way a person would:
// tap, walk (the phone's position is moved), type readings. Prints what happened as JSON.
//
//   node run_visit.js <app url> <truth.json> <out dir>
//
// truth.json holds {"truth": [ln(Cd/threshold) per cell], "threshold": x, "kit_sd": s, "seed": n}.
const fs = require("fs");
const path = require("path");
const { chromium, devices } = require("playwright");

function rng(seed) {                       // small deterministic generator with a normal draw
  let s = seed >>> 0;
  const u = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
}

(async () => {
  const [url, truthPath, outDir] = process.argv.slice(2);
  const sim = JSON.parse(fs.readFileSync(truthPath, "utf8"));
  const normal = rng(sim.seed || 1);
  const browser = await chromium.launch();
  const context = await browser.newContext({ ...devices["Pixel 7"], permissions: ["geolocation"], geolocation: { longitude: 0, latitude: 0 }, acceptDownloads: true });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

  await page.goto(url);
  await page.waitForSelector("#threshold");
  const pkg = await page.evaluate(() => JSON.parse(localStorage.getItem("soilsight.pkg")));
  await page.evaluate(() => navigator.serviceWorker.ready);          // the app's files are now kept on the phone

  // ---- from here on there is no network at all
  await context.setOffline(true);
  await page.reload();                                               // opens from the phone's own copy
  await page.waitForSelector("#threshold");
  const offlineStart = await page.evaluate(() => fetch("package.json").then(() => "reached", () => "unreachable"));

  await page.fill("#threshold", String(sim.threshold));
  await page.click("[data-do=start]");
  await page.waitForSelector("#map");
  const shot = (name) => page.screenshot({ path: path.join(outDir, name), fullPage: true });
  await shot("1_started.png");

  const g = pkg.grid, geo = pkg.geo.to_lonlat;
  const cellXY = (c) => [(g.cols[c] + 0.5) * g.cell_m, (g.ny - g.rows[c] - 0.5) * g.cell_m];
  const walkTo = async (x, y) => {
    await context.setGeolocation({ longitude: geo[0] * x + geo[1] * y + geo[2], latitude: geo[3] * x + geo[4] * y + geo[5], accuracy: 2.5 });
    await page.waitForFunction(([x, y]) => { const v = window.SoilSightApp; return v && document.getElementById("gps").textContent.indexOf("GPS") === 0 && Math.abs(v.position().x - x) < 0.5 && Math.abs(v.position().y - y) < 0.5; }, [x, y]);
  };
  const nearestCell = (x, y) => { let best = 0, bd = Infinity; for (let i = 0; i < g.rows.length; i++) { const [cx, cy] = cellXY(i), d = (cx - x) ** 2 + (cy - y) ** 2; if (d < bd) { bd = d; best = i; } } return best; };

  const guidance = [];
  let reloaded = false, steps = 0;
  for (;;) {
    const v = await page.evaluate(() => window.SoilSightApp.visit());
    const planned = v.spots.filter((s) => s.state === "planned");
    const sampled = v.spots.filter((s) => s.state === "sampled");
    if (!planned.length && !sampled.length) break;
    if (++steps > 200) throw new Error("the visit did not finish");
    if (planned.length) {                                            // walk to the spot, a couple of metres off, and sample
      const spot = planned[0], [px, py] = cellXY(spot.cell);
      await page.click(`[data-do=select][data-id=${spot.id}]`);
      guidance.push(await page.textContent(".where"));
      const x = px + 1.5 * normal(), y = py + 1.5 * normal();
      await walkTo(x, y);
      guidance.push(await page.textContent(".where"));
      await page.click(`[data-do=sampled][data-id=${spot.id}]`);
      continue;
    }
    // test the samples that are waiting, and type the readings in
    for (const spot of sampled) {
      const s = v.samples[spot.id];
      let ln = 0;                                                    // three cores mixed, a few metres apart
      for (let k = 0; k < pkg.protocol.cores; k++) ln += Math.exp(sim.truth[nearestCell(s.x + 3 * normal(), s.y + 3 * normal())]);
      const reading = sim.threshold * (ln / pkg.protocol.cores) * Math.exp(sim.kit_sd * normal());
      await page.click(`[data-do=enter][data-id=${spot.id}]`);
      await page.fill("#reading", reading.toPrecision(4));
      await page.fill("#ph", (5.2 + 0.3 * normal()).toFixed(1));
      await page.click(`[data-do=save][data-id=${spot.id}]`);
    }
    if (!reloaded) {                                                 // close and reopen the app mid-visit, still offline
      await shot("2_first_results.png");
      await page.reload();
      await page.waitForSelector("#map");
      reloaded = true;
    }
  }
  await shot("3_all_tests.png");
  await page.click("[data-do=finish]");
  await shot("4_finished.png");
  const [download] = await Promise.all([page.waitForEvent("download"), page.click("[data-do=exportLog]")]);
  const logPath = path.join(outDir, "log.json");
  await download.saveAs(logPath);
  const zones = await page.evaluate(() => window.SoilSightApp.zones());
  const visit = await page.evaluate(() => window.SoilSightApp.visit());
  await browser.close();
  process.stdout.write(JSON.stringify({ errors, offlineStart, guidance, zones, tests: visit.tests.length, spots: visit.spots.length, log: logPath, download: download.suggestedFilename() }));
})().catch((e) => { console.error(e); process.exit(1); });
