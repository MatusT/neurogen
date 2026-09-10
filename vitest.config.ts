import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // GC-retention regression tests need to force a real collection.
    pool: "forks",
    poolOptions: { forks: { execArgv: ["--expose-gc"] } },
    // Otherwise picks up the same test files again from every git worktree
    // nested under .claude/worktrees/ (each is a parallel task checkout).
    exclude: ["**/node_modules/**", "**/.claude/worktrees/**"],
  },
});
