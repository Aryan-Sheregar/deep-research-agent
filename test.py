import os
from dotenv import load_dotenv
from agent_tools import (scrape_website, get_web_search_results)
from knowledge_base import (
    get_model_embeddings,
    get_text_splitter,
    get_vector_store,
    add_context_to_vector_store,
)

load_dotenv()

search_tool = get_web_search_results()
embeddings = get_model_embeddings()
text_splitter = get_text_splitter()
vector_store = get_vector_store(embeddings)

search_res = search_tool.invoke({"query": "Latest advancements in AI"})
print("Search Results:")
for res in search_res:
    print(f"- {res}")

for result in search_res['results']:
    url = result.get('url')
    print(f"\nScraping URL: {url}")
    scraped_text = scrape_website(url)
    if scraped_text and not scraped_text.startswith("Error"):
        add_context_to_vector_store(
            vector_store, text_splitter, scraped_text, url)
        print("Scaping and addition to vector store successful.")
    else:
        print(f"Failed to scrape or no content found for {url}")


# test_url = "https://arxiv.org/pdf/2003.13461"
# scraped_text = scrape_website(test_url)
# if scraped_text.startswith("Error"):
#     print("Failed to scrape the website.")
# else:
#     print("Website scraped successfully. Here's a preview:")
#     print(scraped_text[:500])  # Print the first 500 characters of the scraped text
