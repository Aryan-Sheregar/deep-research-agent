# Deep Research Agent 

This project is an advanced, autonomous research agent built with Python, Streamlit, and LangChain. It mimics the functionality of deep research features found in models like Gemini, allowing a user to perform comprehensive research on any given topic. The agent autonomously discovers, extracts, indexes, and synthesizes information from various web sources to produce a structured, citable report.



---
## Key Features

- **Autonomous Research Sessions:** Start a new research session on any topic, which automatically clears the old context and knowledge base.
- **Agentic Query Expansion:** Automatically brainstorms and expands a simple topic into multiple, detailed search queries to gather more comprehensive data.
- **Universal Content Extraction:** Intelligently scrapes content from both standard HTML websites and PDF documents.
- **Local Knowledge Base:** Uses a local ChromaDB vector store and Gemma embeddings (via Ollama) to create a private, topic-specific knowledge base for each session.
- **Interactive Chat Interface:** A user-friendly web interface built with Streamlit for asking follow-up questions and refining research.
- **Explainable AI:** The agent's full thought process—including the tools it chooses and the data it observes—is displayed for transparency.
- **Formatted Reporting with Sources:** The final output is a structured Markdown report, complete with a "Sources" section listing the URLs the agent used for its answer.
- **Export Results:** Reports can be downloaded directly from the UI as both Markdown and PDF files.

---
## Technology Stack

- **Frontend:** Streamlit
- **Agent Framework:** LangChain
- **LLM (Reasoning):** Google Gemini API (Gemini 1.5 Flash)
- **Embeddings (Local):** Ollama with `gemma:2b`
- **Web Search Tool:** Tavily Search API
- **Vector Database:** ChromaDB (Local)
- **Web Scraping:** BeautifulSoup4 (HTML), PyMuPDF4LLM (PDF)

---
