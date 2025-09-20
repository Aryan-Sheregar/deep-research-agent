from bs4 import BeautifulSoup
import requests
import fitz
from pymupdf4llm import to_markdown
def scrape_website(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url)
        response.raise_for_status() 
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            doc = fitz.open(stream=response.content, filetype="pdf")
            markdown_text = to_markdown(doc)
            return markdown_text
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            if soup.body:
                text = soup.body.get_text(separator='\n', strip=True)
                return text
            else:
                return "Couldn't find body content."
             
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
