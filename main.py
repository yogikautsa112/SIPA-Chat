from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
from serpapi import GoogleSearch
from deep_translator import GoogleTranslator
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
SERP_API_KEY = os.getenv("SERAPI_KEY")

bert_id = "goy2/bert-model-SIPA-V1"
t5_id = "goy2/t5-squad2-checkpoint"

bert_tokenizer = BertTokenizer.from_pretrained(bert_id)
bert_model = BertForSequenceClassification.from_pretrained(bert_id)
classifier = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer, truncation=True)

t5_tokenizer = T5Tokenizer.from_pretrained(t5_id)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_id)

labels = {0: "kata-kasar", 1: "laporan-kekerasan", 2: "psikologi", 3: "data-umum", 4: "jumlah-kdrt"}

app = FastAPI()
class TextInput(BaseModel):
    question: str

def search_context(query: str) -> str:
    try:
        params = {"engine": "duckduckgo", "q": query, "api_key": SERP_API_KEY, "num": 5}
        results = GoogleSearch(params).get_dict()
        texts = [f"{r.get('title', '')}. {r.get('snippet', '')}".strip()
                for r in results.get("organic_results", []) if r.get("snippet")]
        if not texts:
            return "Context not found."
        combined = " ".join(texts)[:2000]
        return GoogleTranslator(source="auto", target="en").translate(combined)
    except Exception as e:
        return f"Error: {e}"

def generate_answer(q: str, ctx: str) -> str:
    if "Error" in ctx or "Context not found" in ctx:
        return "Maaf, aku tidak bisa menemukan informasi yang relevan."
    q_en = GoogleTranslator(source="auto", target="en").translate(q)
    input_ids = t5_tokenizer(f"question: {q_en} context: {ctx}", return_tensors="pt", truncation=True).input_ids
    output = t5_model.generate(input_ids, max_length=128, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    return GoogleTranslator(source="en", target="id").translate(t5_tokenizer.decode(output[0], skip_special_tokens=True))

def predict_intent(text: str) -> str:
    pred = classifier(text[:512])[0]
    idx = int(pred["label"].split("_")[1])
    return labels.get(idx, "lainnya")

@app.post("/chat")
def chat(payload: TextInput):
    try:
        q = payload.question
        ctx = search_context(q)
        ans = generate_answer(q, ctx)
        intent = predict_intent(q)
        return {"question": q, "predicted_intent": intent, "context": ctx, "response": ans, "timestamp": datetime.utcnow()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
