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

A similar import issue was identified in `partials/reference_resolver.py` concerning the `Entity` TypedDict, which is defined in `partials/entity_resolver.py`. The `reference_resolver.py` file was also attempting to import `Entity` using a `try...except ImportError` block with a fallback dummy definition. This approach is problematic for the concatenated single-file deployment used in Open WebUI.

The fix for `partials/reference_resolver.py` involved:

1.  **Removed Fallback Import:** The `try...except ImportError` block, along with the dummy `Entity` class definition, was removed.
2.  **Employing `TYPE_CHECKING` Block:** An `if TYPE_CHECKING:` block was introduced. Inside this block, `Entity` is imported (`from .entity_resolver import Entity`) solely for the benefit of development-time type checkers and linters. This import does not execute at runtime.
3.  **Using Forward References:** Type hints within method signatures that refer to `Entity` were updated to use forward references (e.g., `Dict[str, 'Entity']`). This practice defers the evaluation of the type hint, which is compatible with the `TYPE_CHECKING` approach and situations where a type might not be fully defined at the exact point it's hinted.

This approach ensures that `reference_resolver.py` correctly relies on the `Entity` TypedDict being globally available at runtime (due to `entity_resolver.py` appearing earlier in the concatenation order defined in `generation_config.json`), while still providing support for development environment tools that perform type analysis.
