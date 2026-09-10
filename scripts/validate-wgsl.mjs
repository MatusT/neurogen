// Composes WGSL modules (resolving `// @import <path>` headers, since WGSL
// has no preprocessor), validates each composed entry point with naga, and
// emits the composed source as a generated TS module — the same artifact
// that gets validated is what orchestration code imports and ships.
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";

const SRC_DIR = resolve("src/wgsl");
const GEN_DIR = resolve("src/wgsl/generated");
const VALIDATE_DIR = resolve(".wgsl-build");
const IMPORT_RE = /^\/\/\s*@import\s+(.+)$/;
const ENTRY_RE = /@(compute|vertex|fragment)\b/;

function listWgslFiles(dir) {
  if (!statSync(dir, { throwIfNoEntry: false })) return [];
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "generated") continue;
    const full = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...listWgslFiles(full));
    else if (entry.name.endsWith(".wgsl")) out.push(full);
  }
  return out;
}

// Returns the import chain in dependency-first order, throwing on cycles.
function resolveImports(file, stack = [], seen = new Set(), out = []) {
  if (stack.includes(file)) {
    throw new Error(`import cycle: ${[...stack, file].map((f) => relative(SRC_DIR, f)).join(" -> ")}`);
  }
  if (seen.has(file)) return out;
  seen.add(file);
  stack.push(file);
  const text = readFileSync(file, "utf8");
  for (const line of text.split("\n")) {
    const match = IMPORT_RE.exec(line.trim());
    if (!match) continue;
    resolveImports(resolve(dirname(file), match[1]), stack, seen, out);
  }
  stack.pop();
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
  return resolveImports(file)
    .map((f) => stripImportHeaders(readFileSync(f, "utf8")))
    .join("\n\n");
}

const allFiles = listWgslFiles(SRC_DIR);
const entryFiles = allFiles.filter((f) => ENTRY_RE.test(readFileSync(f, "utf8")));

if (allFiles.length === 0) {
  console.log("wgsl:validate — no .wgsl files yet, nothing to do");
  process.exit(0);
}
if (entryFiles.length === 0) {
  console.log(`wgsl:validate — ${allFiles.length} library file(s), no entry points yet, nothing to compose`);
  process.exit(0);
}

mkdirSync(VALIDATE_DIR, { recursive: true });
mkdirSync(GEN_DIR, { recursive: true });

let failed = false;
for (const file of entryFiles) {
  const rel = relative(SRC_DIR, file);
  let composed;
  try {
    composed = compose(file);
  } catch (err) {
    failed = true;
    console.error(`FAIL ${rel}: ${err.message}`);
    continue;
  }

  const validatePath = join(VALIDATE_DIR, rel);
  mkdirSync(dirname(validatePath), { recursive: true });
  writeFileSync(validatePath, composed);

  try {
    execFileSync("naga", [validatePath], { stdio: "pipe" });
  } catch (err) {
    failed = true;
    console.error(`FAIL ${rel}`);
    console.error(err.stdout?.toString() ?? err.message);
    console.error(err.stderr?.toString() ?? "");
    continue;
  }

  const genPath = join(GEN_DIR, rel.replace(/\.wgsl$/, ".wgsl.ts"));
  mkdirSync(dirname(genPath), { recursive: true });
  writeFileSync(genPath, `export default ${JSON.stringify(composed)};\n`);
  console.log(`ok   ${rel}`);
}

if (failed) process.exit(1);
