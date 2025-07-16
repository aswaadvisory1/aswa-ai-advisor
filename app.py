from flask import Flask, request, jsonify
import faiss
import numpy as np
import json
import openai
import langdetect
import requests
from bs4 import BeautifulSoup

# === Flask App Setup ===
app = Flask(__name__)
openai.api_key = "sk-Y47oVaZxsKQ-SxYfgSKLFdodxOqrq9bo3oY-Xlz7X-T3BlbkFJaQITxOE96pXEuIo7yvx2MWdsvw3fMz747_zna3NYUA"

# === Load Files on Startup ===
faiss_index = faiss.read_index("faiss_index.faiss")
metadata = json.load(open("consolidated_metadata.json", encoding="utf-8"))
faiss_map = json.load(open("faiss_to_metadata_mapping.json", encoding="utf-8"))
products = [json.loads(line) for line in open("aswa_products.jsonl", encoding="utf-8")]

# === Utility Functions ===
def detect_lang(text):
    try:
        return "ms" if langdetect.detect(text) in ("ms", "id") else "en"
    except:
        return "en"

def get_embedding(text):
    resp = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return np.array(resp["data"][0]["embedding"]).reshape(1, -1)

def search_quran(query):
    try:
        r = requests.get("https://api.quran.com/api/v4/search", params={"q": query})
        return r.json()["search"]["results"][0]["text"]
    except:
        return None

def search_hadith(query):
    try:
        soup = BeautifulSoup(requests.get(f"https://sunnah.com/search?q={query}").text, "html.parser")
        return soup.select_one(".hadith_text").get_text(strip=True)
    except:
        return None

def get_faiss_results(embedding):
    D, I = faiss_index.search(embedding, 5)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        key = faiss_map.get(str(idx))
        if key and key in metadata:
            results.append({**metadata[key], "distance": float(dist)})
    return results

def suggest_products(question):
    keywords = question.lower().split()
    return [
        p for p in products
        if any(word in (p.get("Product Name", "") + " " + p.get("Description", "")).lower() for word in keywords)
    ][:3]

# === Flask Route ===
@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    lang = detect_lang(question)
    embedding = get_embedding(question)
    faiss_results = get_faiss_results(embedding)

    primary = faiss_results[0]["content"] if faiss_results and faiss_results[0]["distance"] < 0.3 else None
    verse = search_quran(question) or search_hadith(question)
    product_suggestions = suggest_products(question)

    prompt = f"""
You are an AI Islamic Financial Advisor from ASWA Advisory.
Respond in {'Bahasa Malaysia' if lang == 'ms' else 'English'} and follow Shariah-compliant tone.

User's question: {question}

{('Berdasarkan maklumat dalaman kami:' if lang == 'ms' else 'Based on our internal knowledge:') if primary else 'No internal data found.'}

{primary or ''}

Suggested Products:
{'; '.join(f"{p['Product Name']} ({p['Company Name']})" for p in product_suggestions) or 'No relevant products found.'}

Quran or Hadith:
{verse or 'No relevant Quran verse or Hadith found.'}
""".strip()

    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a Shariah-compliant Islamic Financial Advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=700
    )

    return jsonify({
        "response": gpt_response["choices"][0]["message"]["content"].strip(),
        "language": lang,
        "faiss_used": bool(primary),
        "products": product_suggestions,
        "verse": verse,
        "faiss_matches": faiss_results[:3]
    })

# === Local Testing ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
