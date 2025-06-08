# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the MinionS/Minions OpenWebUI project - an implementation of HazyResearch's Minion and MinionS protocols as Functions for Open WebUI. It enables sophisticated collaboration between local AI models (e.g., Ollama) and powerful remote AI models (e.g., Claude) to efficiently process documents and answer complex queries.

## Common Commands

### Function Generation
```bash
# Generate default Minion function (conversational protocol)
python generator_script.py minion

# Generate default MinionS function (task decomposition protocol)
python generator_script.py minions

# Generate with custom profile
python generator_script.py minion --profile custom_profile_name
```

### Development Workflow
1. Modify partials in the `partials/` directory
2. Run the generator script to create updated functions
3. Copy generated functions from `generated_functions/` to Open WebUI

## Architecture Overview

### Modular Design
The project uses a modular architecture where complete functions are assembled from "partials":

- **Generator Script** (`generator_script.py`): Assembles partials based on configuration
- **Configuration** (`generation_config.json`): Defines profiles and partial combinations
- **Partials** (`partials/`): Modular code components that are combined to create functions

### Key Components

1. **Valves** (`*_valves.py`): Configuration settings including API keys, model selection, timeouts
2. **Models** (`*_models.py`): Pydantic models defining data structures
3. **Prompts** (`*_prompts.py`): Logic for generating prompts for AI models
4. **Protocol Logic** (`*_protocol_logic.py`): Core implementation of Minion/MinionS protocols
5. **Pipe Method** (`*_pipe_method.py`): Integration point with Open WebUI's pipe system

### Protocol Differences
- **Minion**: Conversational approach where Claude interviews Ollama about document content
- **MinionS**: Task decomposition where Claude breaks queries into sub-tasks executed by Ollama

### Performance Features (v0.3.4b)
- Smart information sufficiency scoring
- Dynamic convergence detection
- Adaptive thresholds based on document/query complexity
- Performance profiles: high_quality, balanced, fastest_results
- Token savings analysis and reporting

## Important Notes

- Functions must be regenerated after modifying partials
- Generated functions go to `generated_functions/` directory
- The project currently lacks automated testing - be careful with changes
- Debug mode can be enabled via valves for troubleshooting
- API keys must be configured in Open WebUI's function settings