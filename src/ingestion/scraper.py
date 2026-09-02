import json, time, sys, requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DIR, SCRAPE_URLS, MAX_PAGES, SCRAPED_DOCS_PATH

def is_valid_url(url, base_domain):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"): return False
    if base_domain not in parsed.netloc: return False
    skip = [".pdf", ".png", ".jpg", ".gif", "#", "mailto:", "javascript:"]
    return not any(s in url.lower() for s in skip)

def extract_text(html, url):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title else "Untitled"
    main = (soup.find("main") or soup.find("article")
            or soup.find(id=lambda i: i and "content" in i.lower() if i else False)
            or soup.find("body"))
    raw = main.get_text(separator=" ", strip=True) if main else ""
    text = " ".join(raw.split())
    return {"url": url, "title": title, "text": text, "char_count": len(text)}

def crawl(seed_urls, max_pages=MAX_PAGES):
    visited, queue, documents = set(), list(seed_urls), []
    headers = {"User-Agent": "Mozilla/5.0 (educational-research-bot)"}
    pbar = tqdm(total=max_pages, desc="Scraping pages")
    while queue and len(documents) < max_pages:
        url = queue.pop(0)
        if url in visited: continue
        visited.add(url)
        base_domain = urlparse(url).netloc
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200: continue
            doc = extract_text(resp.text, url)
            if doc["char_count"] > 200:
                documents.append(doc); pbar.update(1); pbar.set_postfix({"url": url[-55:]})
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                if href not in visited and is_valid_url(href, base_domain):
                    queue.append(href)
            time.sleep(0.5)
        except Exception as exc:
            print(f"\nSkipping {url}: {exc}")
    pbar.close()
    return documents

def save_documents(documents):
    with open(SCRAPED_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(documents)} docs to {SCRAPED_DOCS_PATH}")
    return SCRAPED_DOCS_PATH

def load_documents():
    with open(SCRAPED_DOCS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
