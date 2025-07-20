# neetcode_scraper.py
import requests
from bs4 import BeautifulSoup
import json

def scrape_neetcode(tab="neetcode250"):
    url = f"https://neetcode.io/practice?tab={tab}"
    res = requests.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    questions = []
    for a in soup.select("a.practice-card"):
        title = a.select_one(".card-title").get_text(strip=True)
        difficulty = a.select_one(".difficulty").get_text(strip=True)
        link = a["href"]
        full_link = f"https://neetcode.io{link}"
        questions.append({"title": title, "difficulty": difficulty, "url": full_link})
    
    with open("neet_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    return questions

if __name__ == "__main__":
    qs = scrape_neetcode()
    print(f"Scraped {len(qs)} questions.")
