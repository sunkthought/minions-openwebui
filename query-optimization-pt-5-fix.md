# Query Optimization & Reformulation - Part 5 Import Fix (v0.3.5)

This document details a critical fix applied after the initial Iteration 5 ("Entity and Reference Resolution Framework") submission.

## Issue Encountered

After generating the combined Open WebUI function using `generator_script.py`, an `ImportError` (or a related parsing error like `Cannot parse: 652:0: except ImportError:`) would occur when Open WebUI attempted to load the function.

The root cause was that `partials/query_analyzer.py` was modified to include direct Python imports for `EntityResolver` (from `partials.entity_resolver`) and `ReferenceResolver` (from `partials.reference_resolver`). While these imports work in a standard Python environment where files are separate modules, they fail in the Open WebUI context. In Open WebUI, all partials are concatenated into a single Python script. Thus, the concept of importing from other "partial" files doesn't apply; the necessary classes should already be defined in the global scope by the time they are needed, thanks to the order specified in `generation_config.json`.

The `try...except ImportError` blocks with dummy class definitions, while intended as fallbacks, were still part of this incorrect import logic for the concatenated file context.

## Resolution

The fix involved modifying `partials/query_analyzer.py`:

1.  **Removed Direct Imports:** All `import ... from .entity_resolver` and `import ... from .reference_resolver` lines were removed.
2.  **Removed Fallback Dummy Classes:** The `try...except ImportError` blocks containing dummy definitions for `EntityResolver` and `ReferenceResolver` were removed from `query_analyzer.py`.
3.  **Reliance on Concatenation Order:** The `QueryAnalyzer` now directly uses `EntityResolver` and `ReferenceResolver` (e.g., `self.entity_resolver = EntityResolver(...)`). This is correct because `generation_config.json` ensures that `entity_resolver.py` and `reference_resolver.py` are included *before* `query_analyzer.py` in the final generated script, making these classes available in the global scope.

This change ensures that the generated function script is valid and does not attempt to perform relative imports that are incompatible with the single-file deployment model in Open WebUI.
