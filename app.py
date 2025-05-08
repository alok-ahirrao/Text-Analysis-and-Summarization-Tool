from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
import torch
from scipy.special import softmax
from collections import defaultdict
import PyPDF2
import io
import time
import requests
import wikipedia
from urllib.parse import quote
import json
from datetime import datetime, timedelta
import os # Added import

app = Flask(__name__)

# Load summarization model and tokenizer
sum_model_name = "t5-base"
sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)

# Load sentiment analysis model and tokenizer
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# Load NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

# Load text classification pipeline for categories
category_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli")

# NewsAPI configuration
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")  # Replaced hardcoded key
NEWSAPI_URL = "https://newsapi.org/v2/everything"

def detect_category(text):
    # Define possible categories
    categories = ["Politics", "Technology", "Sports", "Business", "Entertainment", "Science", "Health"]
    
    # Check text against each category
    results = []
    for category in categories:
        output = category_pipeline(f"This text is about {category.lower()}: {text[:512]}")
        results.append((category, output[0]['score']))
    
    # Sort by score and get top categories
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]  # Return top 3 categories with scores

def get_wikipedia_summary(topic, max_sentences=3):
    """Get a summary from Wikipedia for a given topic."""
    try:
        # Search Wikipedia
        search_results = wikipedia.search(topic, results=3)
        if not search_results:
            return None
        
        # Try to get the most relevant page
        for result in search_results:
            try:
                # Get summary and URL
                summary = wikipedia.summary(result, sentences=max_sentences, auto_suggest=False)
                page = wikipedia.page(result, auto_suggest=False)
                return {
                    'title': result,
                    'summary': summary,
                    'url': page.url
                }
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                continue
            except Exception as e:
                print(f"Error getting Wikipedia summary for {result}: {e}")
                continue
    except Exception as e:
        print(f"Error searching Wikipedia: {e}")
        return None
    return None

def analyze_related_topics(text):
    """Extract topics and find related information using Wikipedia."""
    try:
        # Use the summarization model to extract key topics
        inputs = sum_tokenizer("extract key topics: " + text[:512], return_tensors="pt", max_length=512, truncation=True)
        topics_ids = sum_model.generate(
            inputs["input_ids"],
            max_length=100,
            min_length=30,
            num_beams=4,
            early_stopping=True
        )
        topics = sum_tokenizer.decode(topics_ids[0], skip_special_tokens=True)
        
        # Split into topics and clean them
        key_phrases = [topic.strip() for topic in topics.split('.') if topic.strip()]
        
        # Get Wikipedia information for each topic
        related_info = []
        for phrase in key_phrases[:3]:  # Limit to top 3 topics
            wiki_info = get_wikipedia_summary(phrase)
            if wiki_info:
                related_info.append({
                    'topic': phrase,
                    'details': [{
                        'title': wiki_info['title'],
                        'summary': wiki_info['summary'],
                        'link': wiki_info['url']
                    }]
                })
        
        return related_info
    except Exception as e:
        print(f"Error in topic analysis: {e}")
        return []

def generate_summary(text):
    inputs = sum_tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = sum_model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def analyze_sentiment(text):
    # Truncate text if too long for the sentiment model
    inputs = sentiment_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    scores = outputs.logits[0].detach().numpy()
    scores = softmax(scores)
    
    # Get labels from model config
    labels = sentiment_model.config.id2label
    sentiment_scores = {labels[i]: float(scores[i]) for i in range(len(scores))}
    
    # Determine dominant sentiment
    dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    
    return dominant_sentiment, sentiment_scores

def extract_entities(text):
    try:
        # Split text into smaller chunks to avoid overloading the model
        chunk_size = 500
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        all_entities = []
        for chunk in chunks:
            chunk_entities = ner_pipeline(chunk)
            all_entities.extend(chunk_entities)
        
        if not all_entities:
            return {}

        # Group entities by type
        grouped_entities = defaultdict(list)
        seen_entities = defaultdict(set)  # Track unique entities per type
        
        for entity in all_entities:
            word = entity['word'].replace('##', '')
            entity_type = entity['entity_group']
            score = round(entity['score'], 3)
            
            # Only add if we haven't seen this entity for this type
            if word.lower() not in seen_entities[entity_type]:
                seen_entities[entity_type].add(word.lower())
                grouped_entities[entity_type].append({
                    'word': word,
                    'score': score
                })
        
        # Sort entities by score and filter empty groups
        final_entities = {}
        for entity_type, entities_list in grouped_entities.items():
            if entities_list:
                sorted_entities = sorted(entities_list, key=lambda x: x['score'], reverse=True)
                final_entities[entity_type] = sorted_entities[:5]  # Limit to top 5 entities per type
        
        return final_entities
    except Exception as e:
        print(f"Error during entity extraction: {e}")
        return {}

def search_related_news(entities, max_results=5):
    """Search news using NewsAPI based on entities."""
    if not entities or not NEWSAPI_KEY:
        return []

    try:
        # Build search query from entities
        keywords = []
        priority_types = ['ORG', 'PERSON', 'GPE']
        
        # First try priority entities
        for entity_type in priority_types:
            if entity_type in entities:
                keywords.extend([entity['word'] for entity in entities[entity_type][:2]])
        
        # If no priority entities, use any entities
        if not keywords:
            for entity_list in entities.values():
                keywords.extend([entity['word'] for entity in entity_list[:1]])
                if len(keywords) >= 2:
                    break
        
        if not keywords:
            return []

        # Create search query
        query = " OR ".join(set(keywords))
        
        # Set date range (last 30 days)
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Make API request
        params = {
            'q': query,
            'apiKey': NEWSAPI_KEY,
            'language': 'en',
            'sortBy': 'relevancy',
            'from': from_date,
            'to': to_date,
            'pageSize': max_results
        }
        
        response = requests.get(NEWSAPI_URL, params=params)
        response.raise_for_status()
        news_data = response.json()
        
        if news_data['status'] != 'ok':
            print(f"NewsAPI error: {news_data.get('message', 'Unknown error')}")
            return []
        
        # Format results
        formatted_results = []
        for article in news_data.get('articles', [])[:max_results]:
            formatted_results.append({
                'title': article['title'],
                'url': article['url'],
                'source': article['source']['name'],
                'date': article['publishedAt'].split('T')[0]
            })
        
        return formatted_results
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []
    except Exception as e:
        print(f"Error processing news results: {e}")
        return []

@app.route("/", methods=["GET", "POST"])
def home():
    summary = None
    original_text = None
    sentiment = None
    sentiment_details = None
    entities = None
    related_news = None
    categories = None
    related_topics = None
    error_message = None
    input_text = None # Variable to hold the text to be analyzed

    if request.method == "POST":
        pdf_file = request.files.get('pdf_file')
        pasted_text = request.form.get('text_input')

        # --- Determine Input Source ---
        if pdf_file and pdf_file.filename != '':
            if pdf_file.filename.endswith('.pdf'):
                try:
                    pdf_stream = io.BytesIO(pdf_file.read())
                    pdf_reader = PyPDF2.PdfReader(pdf_stream)
                    extracted_text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text: # Check if text extraction returned something
                             extracted_text += page_text + "\n"

                    if extracted_text.strip():
                        input_text = extracted_text
                    else:
                        error_message = "Could not extract text from the PDF. It might be image-based, password-protected, or corrupted."
                except Exception as e:
                    print(f"Error processing PDF: {e}")
                    error_message = f"An error occurred processing the PDF: {e}"
            else:
                error_message = "Invalid file type. Please upload a PDF."
        elif pasted_text and pasted_text.strip() != '':
            input_text = pasted_text
        else:
            # Neither PDF nor text was provided (or PDF was empty filename)
            error_message = "Please upload a PDF or paste text into the text box."

        # --- Perform Analysis if Input Text is Available ---
        if input_text and not error_message:
            original_text = input_text # Keep a copy for display
            try:
                summary = generate_summary(input_text)
                sentiment, sentiment_details = analyze_sentiment(input_text)
                entities = extract_entities(input_text)
                related_news = search_related_news(entities)
                categories = detect_category(input_text)
                related_topics = analyze_related_topics(input_text)
            except Exception as e:
                print(f"Error during analysis: {e}")
                error_message = f"An error occurred during text analysis: {e}"
                # Clear results if analysis fails
                summary = None
                sentiment = None
                entities = None
                related_news = None
                categories = None
                related_topics = None


    return render_template("index.html",
                           summary=summary,
                           original_text=original_text, # Pass original text for display
                           sentiment=sentiment,
                           entities=entities,
                           related_news=related_news,
                           error_message=error_message,
                           categories=categories,
                           related_topics=related_topics)
