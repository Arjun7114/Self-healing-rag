\# Self-Healing RAG



A Retrieval-Augmented Generation (RAG) system that checks and retries its own

answers instead of trusting the first result. If an answer isn't supported by the

retrieved sources, it searches again with a better query — or honestly replies

"I don't have enough information" instead of making something up.



Built with Python and LangGraph.



\## Status

Work in progress — building in phases.

