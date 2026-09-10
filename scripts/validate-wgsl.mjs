// Composes WGSL modules (resolving `// @import <path>` headers, since WGSL
// has no preprocessor) and validates each composed entry point with naga.
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";

const SRC_DIR = resolve("src/wgsl");
const OUT_DIR = resolve(".wgsl-build");
const IMPORT_RE = /^\/\/\s*@import\s+(.+)$/;

function listWgslFiles(dir) {
  if (!statSync(dir, { throwIfNoEntry: false })) return [];
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...listWgslFiles(full));
    else if (entry.name.endsWith(".wgsl")) out.push(full);
  }
  return out;
}

function resolveImports(file, seen = new Set(), out = []) {
  if (seen.has(file)) return out;
  seen.add(file);
  const text = readFileSync(file, "utf8");
  for (const line of text.split("\n")) {
    const match = IMPORT_RE.exec(line.trim());
    if (!match) continue;
    const importPath = resolve(dirname(file), match[1]);
    resolveImports(importPath, seen, out);
  }
  out.push(file);
  return out;
}

function stripImportHeaders(text) {
  return text
    .split("\n")
    .filter((line) => !IMPORT_RE.test(line.trim()))
    .join("\n");
}

function compose(file) {
  const chain = resolveImports(file);
  return chain.map((f) => stripImportHeaders(readFileSync(f, "utf8"))).join("\n\n");
}

const entryFiles = listWgslFiles(SRC_DIR);
if (entryFiles.length === 0) {
  console.log("wgsl:validate — no .wgsl files yet, nothing to do");
  process.exit(0);
}

mkdirSync(OUT_DIR, { recursive: true });

let failed = false;
for (const file of entryFiles) {
  const composed = compose(file);
  const outFile = join(OUT_DIR, relative(SRC_DIR, file));
  mkdirSync(dirname(outFile), { recursive: true });
  writeFileSync(outFile, composed);
  try {
    execFileSync("naga", [outFile], { stdio: "pipe" });
    console.log(`ok   ${relative(SRC_DIR, file)}`);
  } catch (err) {
    failed = true;
    console.error(`FAIL ${relative(SRC_DIR, file)}`);
    console.error(err.stdout?.toString() ?? err.message);
    console.error(err.stderr?.toString() ?? "");
  }
}

if (failed) process.exit(1);
