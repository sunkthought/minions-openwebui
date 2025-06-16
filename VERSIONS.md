# Version History

## v0.3.9b - Enhanced Web Search & Granular Streaming (Latest)
- **🔍 Complete Web Search Tool Execution**: Full implementation of Open WebUI's tool execution lifecycle
  - **Tool Response Processing**: Comprehensive handling of Open WebUI's search tool responses with multiple format support
  - **Response Normalization**: Automatic normalization of various search provider formats (Google, Bing, DuckDuckGo)
  - **Tool Execution Bridge**: New `ToolExecutionBridge` class for seamless integration with Open WebUI's async tool system
  - **Error Recovery**: Robust fallback handling with configurable retry logic and timeout management
- **📊 Granular Streaming Updates**: Fine-grained progress tracking within execution phases
  - **Sub-Phase Progress**: Detailed progress within each major phase (0-100% per phase)
  - **Task Decomposition Progress**: Step-by-step updates for query analysis, document structure, task generation
  - **Chunk-Level Progress**: Individual progress tracking for each task-chunk combination
  - **Conversation Round Progress**: Detailed updates for each conversation round in Minion protocol
  - **Visual Progress Bars**: Text-based progress bars with percentage indicators
- **⚡ Rate-Limited Streaming**: Intelligent update throttling to prevent UI flooding
  - **Adaptive Rate Limiting**: Configurable minimum interval between updates (default 0.1s)
  - **Update Queuing**: Automatic queuing of rapid updates with batch delivery
  - **Force Update Support**: Critical updates can bypass rate limiting
- **🛠️ Protocol Enhancement Points**: Strategic integration of progress tracking
  - **MinionS Protocol**: Progress tracking in task decomposition, execution, and synthesis phases
  - **Minion Protocol**: Conversation round progress with stage indicators
  - **Web Search Phase**: Query formulation, execution, parsing, and citation progress
  - **Synthesis Phase**: Result collection, answer generation, and formatting progress

## v0.3.9 - Open WebUI Integration Suite
- **🌐 Full Open WebUI Search Integration**: Native web search capabilities using Open WebUI's search tool format
  - **Smart Detection**: Automatic identification of queries requiring external information (current events, fact-checking, latest data)
  - **Tool Format Compliance**: Uses `__TOOL_CALL__` format for seamless integration with Open WebUI's search infrastructure
  - **Task Type Classification**: Intelligent categorization as document_analysis, web_search, or hybrid approaches
  - **Context Enhancement**: Real-time web information enriches both conversational and task-based responses
- **🔍 Native RAG Pipeline Integration**: Full leverage of Open WebUI's RAG infrastructure instead of naive chunking
  - **Document Reference Support**: Complete implementation of `#document_name` syntax for targeted retrieval
  - **Intelligent Retrieval**: Advanced relevance scoring with configurable thresholds (rag_relevance_threshold)
  - **Multi-Document Operations**: Seamless cross-document analysis in knowledge base environments
  - **Smart Fallback**: Graceful degradation to chunking when RAG infrastructure is unavailable
- **📚 Advanced Citation System**: Native Open WebUI inline citation format with comprehensive source tracking
  - **Inline Citation Tags**: Full support for `<cite title="source">text</cite>` format
  - **Multi-Source Citations**: Unified citation handling for documents, web searches, and RAG retrievals
  - **Automatic Attribution**: Intelligent matching of responses to source materials with relevance scoring
  - **Source Traceability**: Complete audit trail from response text back to original sources
- **🔄 Streaming Response Support**: Real-time progress updates during long-running operations
  - **AsyncGenerator Architecture**: Built on Python's AsyncGenerator for efficient streaming implementation
  - **Phase-by-Phase Updates**: Live progress for query analysis, task decomposition, execution, and synthesis
  - **Error-Aware Streaming**: Graceful error handling within streaming context with user-friendly messages
  - **Protocol-Specific Streaming**: Tailored streaming for both conversational (Minion) and task-based (MinionS) flows
- **📊 Visual Task Decomposition (MinionS)**: Real-time task visualization using Mermaid diagrams
  - **Dynamic Diagram Generation**: Live Mermaid syntax generation showing task relationships and data flow
  - **Status-Aware Visualization**: Color-coded task status (pending, running, completed, failed) with real-time updates
  - **Dependency Mapping**: Clear visual representation of task dependencies and execution order
  - **Progress Integration**: Synchronized updates between streaming progress and visual representation
- **🗄️ Multi-Document Knowledge Base Support**: Advanced cross-document analysis and conversation capabilities
  - **Document Registry**: Automatic tracking of available documents with comprehensive metadata
  - **Cross-Document Relationships**: Detection and utilization of relationships between documents
  - **Conversation Context**: Multi-document awareness in conversational flows (Minion protocol)
  - **Knowledge Base Operations**: Intelligent analysis across multiple documents simultaneously
- **⚙️ Universal Backward Compatibility**: All features include toggle valves with sensible defaults
  - **Feature Toggles**: Individual control over each v0.3.9 feature for gradual adoption
  - **Default Settings**: Conservative defaults ensure compatibility with existing deployments
  - **Protocol-Specific Features**: Tailored feature sets for Minion (conversational) vs MinionS (task-based) protocols

## v0.3.8 - Multi-Provider API Support & Intelligent Scaling
- **🔄 Multi-Provider API Support**: Added OpenAI API support as an alternative to Anthropic Claude
  - **Provider Selection**: Choose between 'anthropic' or 'openai' for the supervisor model via `supervisor_provider` valve
  - **Unified API Interface**: `call_supervisor_model()` function provides seamless switching between providers
  - **Provider-Specific Configuration**: Separate API keys and model selection for each provider
  - **Model Support**: Compatible with GPT-4o, GPT-4-turbo, Claude-3.5-Sonnet, and other models
- **📈 Scaling Strategies (MinionS)**: Implemented three scaling strategies from the HazyResearch paper
  - **Repeated Sampling**: Execute tasks multiple times and aggregate results for higher confidence
  - **Finer Decomposition**: Break down complex tasks into smaller, more focused sub-tasks
  - **Context Chunking**: Process large documents by intelligently splitting content across chunks
  - **Strategy Selection**: Configure via `scaling_strategy` valve with options: none, repeated_sampling, finer_decomposition, context_chunking
- **🧠 Adaptive Round Control**: Intelligent stopping based on information gain analysis
  - **Information Gain Tracking**: Monitor the novelty and value of information gathered each round
  - **Dynamic Convergence**: Automatically detect when additional rounds provide diminishing returns
  - **Configurable Thresholds**: Fine-tune sensitivity with `min_info_gain` setting (0.0-1.0)
  - **Smart Termination**: Balance thoroughness with efficiency by stopping when sufficient information is gathered
- **🔍 Model Capability Detection**: Automatic parameter adjustment based on model capabilities
  - **Capability Database**: Comprehensive database of model capabilities (context limits, JSON support, function calling)
  - **Automatic Detection**: Detect model capabilities for both supervisor and worker models
  - **Dynamic Adjustment**: Automatically adjust chunk sizes, token limits, and processing strategies
  - **Performance Optimization**: Optimize processing based on each model's strengths and limitations
- **🐛 Critical Bug Fixes**: Resolved JSON parsing errors that prevented local model execution
  - **Escape Sequence Handling**: Fixed invalid JSON escape sequences generated by local models (e.g., `\a`, `\c`)
  - **Robust Parsing**: Improved JSON parsing with better error recovery and fallback mechanisms
  - **Preventive Fixes**: Applied fixes to both Minion and MinionS protocols for consistency

## v0.3.7 - Modular Architecture & Code Quality Improvements
- **🏗️ Modular Architecture**: Complete redesign using centralized utility modules for better maintainability and consistency
  - **Centralized Import Management**: `imports_registry.py` eliminates duplicate imports and organizes dependencies
  - **Constants Extraction**: `constants.py` centralizes 200+ magic numbers, timeouts, model names, and configuration values
  - **Unified Error Handling**: `error_handling.py` provides consistent error formatting, context-aware messages, and troubleshooting hints
  - **Structured Debug Logging**: `debug_utils.py` offers hierarchical context management, timing utilities, and multiple debug levels
  - **Protocol Base Classes**: `protocol_base.py` extracts common patterns between Minion and MinionS for code reuse
  - **State Management**: `protocol_state.py` centralizes round tracking, metrics collection, and execution context
- **🔧 Enhanced Generator Script**: Updated to support both legacy and modular architectures with backward compatibility
- **📊 New v0.3.7 Profiles**: Added `minion_v037` and `minions_v037` profiles leveraging the new modular structure
- **🛠️ Developer Experience**: Improved code organization, consistent patterns, and easier customization through modular design
- **🔄 Backward Compatibility**: Legacy profiles (`minion_default`, `minions_default`) continue to work unchanged
- **📈 Performance Benefits**: Better error recovery, consistent debugging, and reduced code duplication across protocols

## v0.3.6 - Enhanced Minion Protocol
- **Structured Output by Default**: Minion protocol now enables structured output by default for improved reliability and consistency
- **Intelligent Completion Detection**: Added detection of when the remote model has sufficient information using natural language cues like "I now have sufficient information" or "I can now answer"
- **Enhanced Question Generation**: Improved remote model prompting with strategic question guidelines, examples of good vs poor questions, and context-aware tips
- **Better Local Model Prompting**: Enhanced local assistant prompts with clearer role definition, citation guidance, and confidence level criteria
- **Comprehensive Conversation Metrics**: New metrics tracking including round usage, confidence distribution, completion method, duration, and token estimates
- **Document Chunking Support**: Mirrored MinionS chunking capabilities for handling large documents efficiently
  - Automatic document splitting for large files
  - Individual chunk processing with combined results
  - Configurable chunk size and maximum chunks
  - Clear multi-chunk result presentation

## v0.3.5 - Robust Structured Output Support
- **JSON Mode by Default**: Structured output is now enabled by default for improved reliability with compatible local models
- **Model Capability Detection**: Automatically detects if local models support JSON mode (includes llama3.2, mistral, mixtral, qwen2, gemma2, and many others)
- **Enhanced Prompt Engineering**: Clear JSON schema with comprehensive examples of correct and incorrect formats
- **Robust Parsing**: Handles markdown-wrapped JSON, extracts JSON from responses with explanatory text, and includes regex fallback for malformed responses
- **Configurable Fallback**: New `structured_output_fallback_enabled` valve allows control over fallback behavior
- **Success Metrics**: Tracks structured output success rate for monitoring and debugging
- **Better Confidence Scoring**: Consistent confidence values (HIGH/MEDIUM/LOW) mapped to numeric scores for better decision making

## v0.3.4 - Advanced Adaptive Round Management & Performance Insights
- **Smart Information Sufficiency**: Implemented information sufficiency scoring that considers query component coverage and confidence of addressed components, moving beyond simple confidence metrics.
- **Dynamic Convergence Detection**: MinionS can now detect diminishing returns by tracking per-round information gain, novelty of findings, and task failure trends. This allows the system to stop early if further rounds are unlikely to yield significant new information, especially when sufficiency is met.
- **Adaptive Thresholds**: Key decision thresholds (for confidence-based early stopping, sufficiency requirements, and novelty sensitivity in convergence) are now dynamically adjusted based on:
    - Document size (small, medium, large).
    - Query complexity (simple, medium, complex).
    - First-round performance (high novelty can relax subsequent sufficiency needs).
- **Performance Profiles**: Introduced 'high_quality', 'balanced', and 'fastest_results' profiles to allow users to easily tune the base operational parameters (max rounds, base thresholds) before adaptive adjustments.
- **Comprehensive Performance Report**: The final output now includes a detailed report summarizing total rounds, stopping reasons, final sufficiency and convergence scores, and the effective thresholds used during the run.
- **Numerous new valves** added to configure these adaptive behaviors, convergence criteria, and performance profiles.

## v0.3.3 - Adaptive Round Management
- **Smart iteration control**: The system now dynamically adjusts the number of rounds based on task complexity and progress
- **Early termination logic**: Automatically stops when sufficient information is gathered, saving costs
- **Improved efficiency**: Reduces unnecessary API calls while maintaining answer quality

## v0.3.2 - Custom Prompts
- **User-defined prompts**: Added support for custom task instructions and synthesis prompts
- **Enhanced flexibility**: Users can now fine-tune how the supervisor decomposes tasks and synthesizes results
- **Better domain adaptation**: Custom prompts allow optimization for specific document types or query patterns

## v0.3.1 - Task-Specific Instructions and Advanced Prompts
- **Context-aware task generation**: Tasks now include specific instructions based on document content
- **Improved local model guidance**: Better prompting strategies for local models to extract relevant information
- **Enhanced accuracy**: More precise task execution leads to better overall results

## v0.3.0 - Code-Based Task Decomposition
- **Dynamic task generation**: The supervisor now generates Python code to create tasks programmatically
- **Document-aware decomposition**: Tasks are created based on actual document structure and content
- **Scalable approach**: Can handle documents of varying sizes and structures more effectively
- **Improved Minion protocol**: Enhanced conversation flow and better final answer detection

## v0.2.1 - Refactored Architecture
- **Modular design**: Separated concerns into dedicated partials for better maintainability
- **Enhanced error handling**: Improved timeout management and error recovery
- **Better token savings calculation**: More accurate cost estimation

## v0.2.0 - Initial Release
- Basic Minion and MinionS protocol implementation
- Support for Claude and Ollama integration
- Token savings analysis