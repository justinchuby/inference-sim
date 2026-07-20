import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const [siteUrl, outputDirectory] = process.argv.slice(2);
if (siteUrl === undefined || outputDirectory === undefined) {
  throw new Error("usage: preserve-deployed-assets.mjs <site-url> <output-directory>");
}

const site = new URL(siteUrl);
const assetPrefix = `${site.pathname.replace(/\/?$/, "/")}assets/`;
const queue = [site.href];
const visited = new Set();
let preserved = 0;

await mkdir(outputDirectory, { recursive: true });

while (queue.length > 0) {
  const url = queue.shift();
  if (url === undefined || visited.has(url)) continue;
  visited.add(url);

  let response;
  try {
    response = await fetch(url, { cache: "no-store" });
  } catch (error) {
    console.warn(`could not fetch ${url}: ${String(error)}`);
    continue;
  }
  if (!response.ok) {
    console.warn(`could not fetch ${url}: HTTP ${response.status}`);
    continue;
  }

  const bytes = new Uint8Array(await response.arrayBuffer());
  const parsed = new URL(url);
  if (parsed.pathname.startsWith(assetPrefix)) {
    await writeFile(path.join(outputDirectory, path.basename(parsed.pathname)), bytes);
    preserved += 1;
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (!/(html|css|javascript|text)/i.test(contentType)) continue;
  const text = new TextDecoder().decode(bytes);
  const matches = text.matchAll(/(?:https?:\/\/[^"'()\s]+)?\/inference-sim\/assets\/[A-Za-z0-9_.-]+/g);
  for (const match of matches) {
    const discovered = new URL(match[0], site.origin);
    if (discovered.origin === site.origin && discovered.pathname.startsWith(assetPrefix)) {
      queue.push(discovered.href);
    }
  }
  if (parsed.pathname.startsWith(assetPrefix)) {
    const relativeMatches = text.matchAll(/(?:\.\/|assets\/)[A-Za-z0-9_.-]+\.(?:js|css|map)/g);
    for (const match of relativeMatches) {
      const discovered = match[0].startsWith("assets/")
        ? new URL(match[0], site)
        : new URL(match[0], parsed);
      if (discovered.pathname.startsWith(assetPrefix)) queue.push(discovered.href);
    }
  }
}

console.log(`preserved ${preserved} deployed assets`);
