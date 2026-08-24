import urllib.request, json, os, sys

wiki_dir = "/home/slava/fuga/text_corpus/forum"
os.makedirs(wiki_dir, exist_ok=True)

api = "https://ru.wikipedia.org/w/api.php"
articles = []
topics = ["искусственный_интеллект", "программирование", "физика", "математика",
          "Философия", "История_России", "Литература", "Музыка", "Космос",
          "Биология", "Химия", "Экономика", "Психология", "Спорт",
          "Робототехника", "Машинное_обучение", "Вселенная", "Человек",
          "Наука", "Технология", "Искусство", "Культура", "Образование"]

for topic in topics:
    try:
        url = f"{api}?action=query&prop=extracts&exintro=1&explaintext=1&titles={topic}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Fuga/1.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=5).read())
        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if "extract" in page and len(page["extract"]) > 200:
                articles.append(f"=== {page.get('title', topic)} ===\n{page['extract']}\n")
    except:
        pass

if articles:
    with open(os.path.join(wiki_dir, "ru_wikipedia.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(articles))
    print(f"  ✓ {len(articles)} articles")
else:
    print("  ✗ no articles")
    sys.exit(1)
