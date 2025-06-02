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

## Additional Fix: `Entity` Type Import in `reference_resolver.py`

Even after initially attempting to resolve import issues for the `Entity` TypedDict in `partials/reference_resolver.py` using an `if TYPE_CHECKING:` block, a parsing error (`Cannot parse: class ReferenceResolver`) persisted when Open WebUI loaded the generated function. This indicated that the `TYPE_CHECKING` block itself, or its interaction with the Open WebUI's function parsing/loading mechanism, was still problematic.

The fix required a further simplification of `partials/reference_resolver.py`:

1.  **Complete Removal of Import Constructs:** The `if TYPE_CHECKING:` block, and any import statement for `Entity` within it (e.g., `from .entity_resolver import Entity`), were completely removed.
2.  **Sole Reliance on Forward References:** The code now exclusively uses forward references (string literals, e.g., `Dict[str, 'Entity']`) for all type hints involving the `Entity` TypedDict.
3.  **Clarifying Comment:** A comment was added at the top of `reference_resolver.py` to explicitly state that the `Entity` TypedDict is defined in `partials/entity_resolver.py` and is expected to be globally available in the final concatenated script due to the specified concatenation order in `generation_config.json`.

This revised approach makes the `partials/reference_resolver.py` file cleaner and relies entirely on the build process (concatenation order) for the `Entity` type to be defined and available in the global scope at runtime. This method is anticipated to be more robust and compatible with the Open WebUI environment's single-file function execution model.
