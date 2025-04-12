from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
from serpapi import GoogleSearch
from deep_translator import GoogleTranslator
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Load ENV
SERP_API_KEY = os.getenv("SERAPI_KEY")
if not SERP_API_KEY:
    raise ValueError("SERAPI_KEY is missing from environment variables.")

# ========== Load Model ==========
tokenizer_t5 = T5Tokenizer.from_pretrained("oke190231/t5-tokenizer-50percent")
model_t5 = T5ForConditionalGeneration.from_pretrained("oke190231/t5-squad2-checkpoint")

tokenizer_bert = BertTokenizer.from_pretrained("oke190231/bert_intent_model")
model_bert = BertForSequenceClassification.from_pretrained("oke190231/bert_intent_model")
classifier = pipeline("text-classification", model=model_bert, tokenizer=tokenizer_bert)

# ========== Intent Mapping ==========
intent_labels = {
    "kata-kasar": 0,
    "laporan-kekerasan": 1,
    "psikologi": 2,
    "data-umum": 3,
    "jumlah-kdrt": 4
}
reverse_labels = {v: k for k, v in intent_labels.items()}

# ========== FastAPI App ==========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://sipa-chat-production.up.railway.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Request Schema ==========
class QuestionInput(BaseModel):
    question: str

# ========== Fungsi Intent ==========
def predict_intent(texts):
    predictions = classifier(texts)
    results = []
    for text, pred in zip(texts, predictions):
        label_index = int(pred["label"].split("_")[1])
        intent = reverse_labels[label_index]
        results.append((text, intent))
    return results

# ========== Fungsi Cari Konteks ==========
def search_context(query):
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERP_API_KEY,
            "num": 5,
            "hl": "id"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        context_list = []

        if "organic_results" in results:
            for r in results["organic_results"]:
                if "snippet" in r:
                    context_list.append(r["snippet"])

        if not context_list:
            for keyword in fallback_contexts:
                if keyword in query.lower():
                    return fallback_contexts[keyword]
            return "Context not found."

        combined_context = " ".join(context_list)
        # Terjemahkan hasil konteks ke Inggris
        translator = GoogleTranslator(source="auto", target="en")
        translated_context = translator.translate(combined_context)
        return translated_context

    except Exception as e:
        return f"Error SerpAPI: {str(e)}"

# ========== Fungsi Jawab Pertanyaan ==========
def generate_answer(question, context):
    """Jawab pertanyaan dengan model T5. Terjemahkan pertanyaan ke Inggris, hasilnya ke Indonesia."""
    if "Error" in context or "Context not found" in context:
        return "Maaf, aku tidak bisa menemukan informasi yang relevan."

    # Terjemahkan pertanyaan ke Inggris
    translated_question = GoogleTranslator(source="auto", target="en").translate(question)

    # Format input untuk T5
    input_text = f"question: {translated_question}  context: {context}"
    input_ids = tokenizer_t5(input_text, return_tensors="pt").input_ids

    # Generate jawaban
    output_ids = model_t5.generate(input_ids, max_length=128)
    english_answer = tokenizer_t5.decode(output_ids[0], skip_special_tokens=True)

    # Terjemahkan jawaban ke Bahasa Indonesia
    indonesian_answer = GoogleTranslator(source="en", target="id").translate(english_answer)
    return indonesian_answer


# ========== Endpoint API ==========
@app.post("/ask")
def ask_question(data: QuestionInput):
    question = data.question
    intent = predict_intent([question])[0][1]
    context = search_context(question)
    answer = generate_answer(question, context)

    return {
        "question": question,
        "intent": intent,
        "context": context,
        "answer": answer
    }