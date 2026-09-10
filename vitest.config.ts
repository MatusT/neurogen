import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // GC-retention regression tests need to force a real collection.
    pool: "forks",
    poolOptions: { forks: { execArgv: ["--expose-gc"] } },
  },
});
