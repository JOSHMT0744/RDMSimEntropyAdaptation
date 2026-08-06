# TestRDM - VS Code Setup Notes

This project (`TestRDM`) is a small test harness that exercises `RDMSim.jar` (from the `RDMNetwork` project) via `Test.java` / `MAPE_KLoop.java`. It has no Eclipse `.project`/`.classpath` metadata, so VS Code treats it as an **unmanaged (invisible) Java project** driven entirely by `.vscode/settings.json` and `.vscode/launch.json`.

## Configuration

**`.vscode/settings.json`** — tells the Java Language Server which external jars to put on the compile classpath:

```json
{
    "java.project.referencedLibraries": [
        "C:/Users/Dell/OneDrive/Documents/journal_paper/RDMSim package/Jar Files/**/*.jar",
        "C:/Users/Dell/OneDrive/Documents/journal_paper/javafx-sdk-21.0.12/lib/**/*.jar"
    ]
}
```

- `Jar Files/*.jar` — `RDMSim.jar` and `json-simple-1.1.jar`, the actual dependencies `Test.java`/`MAPE_KLoop.java` import (`rdm.management.*`, `rdm.network.*`).
- `javafx-sdk-*/lib/*.jar` — required transitively, because `RDMSimulator.displayResults()` (called from `Test.main`) touches `javafx.application.Application`. Without this, compilation fails with `class file for javafx.application.Application not found` and no output is ever produced.

**`.vscode/launch.json`** — runs `Test.main`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "java",
            "name": "Launch TestRDM",
            "request": "launch",
            "mainClass": "Test",
            "vmArgs": "--module-path \"C:/Users/Dell/OneDrive/Documents/journal_paper/javafx-sdk-21.0.12/lib\" --add-modules javafx.controls,javafx.fxml,javafx.graphics,javafx.swing"
        }
    ]
}
```

- `mainClass: "Test"` — `Test.java` has no `package` statement (default package), so the class name is just `Test`, not `Test.Test`.
- `vmArgs` — JavaFX 11+ is modularized and isn't on the JVM's default module path, so it must be added explicitly at runtime (`--module-path` + `--add-modules`). This mirrors the JavaFX setup used in `RDMNetwork`.
- No `classPaths` entry: the debugger auto-computes the runtime classpath from its own compiled output plus `java.project.referencedLibraries`. Adding an explicit `classPaths` list **replaces** that computed classpath instead of extending it, which silently drops the project's own compiled classes and causes `ClassNotFoundException: Test` even though everything compiles fine.

## Gotchas hit during setup

1. **Compiled output isn't in a project-root `bin/` folder.** For this unmanaged-project setup, the Java Language Server writes `.class` files to an internal location under VS Code's workspace storage (`.../workspaceStorage/<hash>/redhat.java/jdt_ws/TestRDM_<id>/bin`), not to `TestRDM/bin`. A `bin/` folder in this project's root is either leftover from an older Eclipse/manual build or nonexistent — it is not where the live output lives, so don't point `classPaths` at it.
2. **Stale compiled classes cause confusing runtime errors.** An old `bin/` folder here once held `.class` files compiled in 2021 against an older version of the `Probe` interface (`getBandwidthConsumption()` returned `int` instead of `double`), producing `NoSuchMethodError` at runtime even though the source and current `RDMSim.jar` matched. If you see a `NoSuchMethodError`/`NoSuchFieldError` that doesn't match the current source, check for and delete stale compiled output before assuming a source or jar problem.
3. **If imports show as unresolved** (`package rdm.management does not exist`), run `Java: Clean Java Language Server Workspace` (Restart and delete) and confirm the jars appear under the **JAVA PROJECTS → Referenced Libraries** view in the Explorer sidebar. If problems persist even after a clean workspace, check whether the Problems panel entries actually come from the Java Language Server (`"source": "Java"` in the raw diagnostic) — the language server's own diagnostics are authoritative; anything without a `source` field may be a stale/orphaned diagnostic collection worth ignoring or clearing via a full `Developer: Reload Window`.

## Running

Open this folder directly in VS Code (not as a subfolder of an already-open workspace), then use **Run and Debug → Launch TestRDM** (or `F5`) — not the editor's inline "Play" button, which bypasses `launch.json` and won't pick up the JavaFX `vmArgs`.
