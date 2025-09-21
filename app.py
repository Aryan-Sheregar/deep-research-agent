__import__("chrusp").load_sqlite()
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
import markdown
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.api_core.exceptions import ResourceExhausted
from agent_tools import get_web_search_results, scrape_website
from knowledge_base import (
    get_text_splitter,
    get_vector_store,
    add_context_to_vector_store,
    safe_clear_vector_store
)

load_dotenv()

st.set_page_config(page_title="Deep Research Agent", layout="wide")
st.title("Deep Research Agent")


def convert_md_to_pdf(md_content):
    html = markdown.markdown(md_content)
    pdf = FPDF()
    pdf.add_page()
    pdf.write_html(html)
    return bytes(pdf.output())


def run_ingestion_pipeline(topic: str):
    with st.spinner(f"Starting ingestion for '{topic}'..."):
        safe_clear_vector_store()
        search_tool = get_web_search_results()
        text_splitter = get_text_splitter()
        vector_store = get_vector_store()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

        st.session_state['source_urls'] = []

        st.write("Step 1: Expanding research topic into sub-queries...")

        expansion_prompt = PromptTemplate.from_template(
            """You are a world-class researcher. Your task is to expand a given research topic into 3 more detailed and specific search queries that cover different key aspects of this topic. Return ONLY the queries, each on a new line, without any numbering or introduction.

            Topic: {topic}

            Expanded Queries:"""
        )
        expansion_chain = expansion_prompt | llm | StrOutputParser()
        expanded_queries_str = expansion_chain.invoke({"topic": topic})
        expanded_queries = [topic] + expanded_queries_str.strip().split('\n')

        st.info("Generated Sub-Queries:")
        for q in expanded_queries:
            st.markdown(f"- `{q}`")

        st.write("Step 2: Discovering relevant sources for all sub-queries...")
        all_search_results = []
        seen_urls = set()

        for query in expanded_queries:
            search_results = search_tool.invoke({"query": query})
            if search_results and 'results' in search_results:
                for result in search_results.get('results', []):
                    url = result.get('url')
                    if url and url not in seen_urls:
                        all_search_results.append(result)
                        seen_urls.add(url)

        if not all_search_results:
            st.error("Discovery failed. No search results found.")
            return

        st.write(f"Found {len(all_search_results)} unique potential sources.")

        for result in all_search_results:
            url = result.get('url')
            st.write(f"Step 3: Extracting & Indexing: {url}")
            content = scrape_website(url)

            if content:
                add_context_to_vector_store(
                    vector_store, text_splitter, content, url)
                st.session_state['source_urls'].append(url)
            else:
                st.warning(f"Could not extract content from {url}. Skipping.")
    st.success("Ingestion complete! You can now ask questions.")


# Sidebar for starting a new research session
with st.sidebar:
    st.header("Research Session")
    research_topic = st.text_input("Enter a new research topic:")
    if st.button("Start New Research"):
        if research_topic:
            st.session_state['research_active'] = True
            st.session_state['messages'] = []
            run_ingestion_pipeline(research_topic)
            st.rerun()
        else:
            st.warning("Please enter a research topic.")

    if 'source_urls' in st.session_state and st.session_state['source_urls']:
        st.header("Researched Sources")
        for url in st.session_state['source_urls']:
            st.markdown(f"- [{url}]({url})")


# Chat UI
if 'research_active' in st.session_state and st.session_state['research_active']:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever()

    web_search_tool = get_web_search_results()
    web_search_tool.description = "A web search tool. Use this ONLY if the local knowledge base does not provide a sufficient answer to the user's query."

    retriever_tool = create_retriever_tool(
        retriever,
        "knowledge_base_retriever",
        "MUST USE FIRST. Searches the local vector store for detailed, topic-specific information. This is your primary tool for answering user questions."
    )

    tools = [web_search_tool, retriever_tool]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                pdf_bytes = convert_md_to_pdf(message["content"])
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Export as Markdown",
                        data=message["content"],
                        file_name="research_report.md",
                        mime="text/markdown",
                        key=f"md_{i}"
                    )
                with col2:
                    st.download_button(
                        label="Export as PDF",
                        data=pdf_bytes,
                        file_name="research_report.pdf",
                        mime="application/pdf",
                        key=f"pdf_{i}"
                    )

    if user_query := st.chat_input("Ask a follow-up question..."):
        st.session_state.messages.append(
            {"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                try:
                    response = agent_executor.invoke({"input": user_query})

                    retrieved_sources = []
                    for action, observation in response.get('intermediate_steps', []):
                        if action.tool == 'knowledge_base_retriever':
                            retrieved_docs = retriever.invoke(
                                action.tool_input)
                            for doc in retrieved_docs:
                                source = doc.metadata.get('source')
                                if source and source not in retrieved_sources:
                                    retrieved_sources.append(source)

                    sources_text = "\n\n".join(
                        [f"- {source}" for source in retrieved_sources])

                    formatting_prompt = PromptTemplate.from_template(
                        """You are a research assistant. Your task is to reformat the following text into a structured, easy-to-read report using Markdown.
                        - A clear and concise title.
                        - An introductory summary paragraph.
                        - Headings for different sections or key findings.
                        - Bullet points to list important details or challenges.
                        - A concluding summary.
                        - Finally, list the sources provided, under a "Sources" heading.

                        Original Text: {original_text}
                        
                        Sources:
                        {sources}

                        Formatted Report:"""
                    )

                    output_parser = StrOutputParser()
                    formatting_chain = formatting_prompt | llm | output_parser
                    final_report = formatting_chain.invoke({
                        "original_text": response['output'],
                        "sources": sources_text
                    })

                    with st.expander("Show agent's thought process"):
                        thoughts = ""
                        for action, observation in response.get('intermediate_steps', []):
                            thoughts += f"**Action:** `{action.tool}`\n\n**Action Input:** `{action.tool_input}`\n\n**Observation:**\n"
                            if action.tool == 'tavily_search' and isinstance(observation, dict):
                                formatted_obs = ""
                                for res in observation.get('results', []):
                                    formatted_obs += f"- **{res.get('title', 'No Title')}**\n  - {res.get('url', 'No URL')}\n"
                                thoughts += formatted_obs if formatted_obs else "No results found."
                            elif action.tool == 'knowledge_base_retriever' and isinstance(observation, list):
                                formatted_obs = ""
                                for doc in observation:
                                    content_preview = (
                                        doc.page_content[:200] + '...') if len(doc.page_content) > 200 else doc.page_content
                                    source = doc.metadata.get('source', 'N/A')
                                    formatted_obs += f"- **Source:** {source}\n  - *Preview:* {content_preview}\n"
                                thoughts += formatted_obs if formatted_obs else "No relevant information found in the knowledge base."
                            else:
                                thoughts += str(observation)
                            thoughts += "\n\n---\n"
                        st.markdown(thoughts)

                    st.markdown(final_report)

                    pdf_bytes_new = convert_md_to_pdf(final_report)
                    col1_new, col2_new = st.columns(2)
                    with col1_new:
                        st.download_button(
                            label="Export as Markdown",
                            data=final_report,
                            file_name="research_report.md",
                            mime="text/markdown",
                            key="md_new"
                        )
                    with col2_new:
                        st.download_button(
                            label="Export as PDF",
                            data=pdf_bytes_new,
                            file_name="research_report.pdf",
                            mime="application/pdf",
                            key="pdf_new"
                        )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_report})

                except ResourceExhausted as e:
                    st.error(
                        "API Quota Exceeded: You've made too many requests to the Gemini API. Please wait a few moments and try again.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I couldn't complete the request due to API rate limits. Please try again shortly."
                    })
else:
    st.info("Start a new research session from the sidebar to begin.")
