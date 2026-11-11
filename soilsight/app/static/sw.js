/* Keeps the app's own files on the phone, so it opens with no signal. Field data never
 * passes through here: the package and the log live in the phone's storage. */
var CACHE = "soilsight-app-v1";
var FILES = ["./", "index.html", "style.css", "app.js", "brain.js", "sha256.js", "manifest.webmanifest", "icon.svg"];

self.addEventListener("install", function (event) {
  event.waitUntil(caches.open(CACHE).then(function (c) { return c.addAll(FILES); }).then(function () { return self.skipWaiting(); }));
});
self.addEventListener("activate", function (event) {
  event.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (k) { return k !== CACHE; }).map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});
self.addEventListener("fetch", function (event) {
  if (event.request.method !== "GET") return;
  event.respondWith(caches.match(event.request, { ignoreSearch: true }).then(function (hit) {
    return hit || fetch(event.request);
  }));
});
