from agent_tools import scrape_website

test_url = "https://arxiv.org/pdf/2003.13461"
scraped_text = scrape_website(test_url)
if scraped_text.startswith("Error"):
    print("Failed to scrape the website.")
else:
    print("Website scraped successfully. Here's a preview:")
    print(scraped_text[:500])  # Print the first 500 characters of the scraped text