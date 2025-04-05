from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import easyocr
import re
import spacy
from transformers import pipeline

app = Flask(__name__)

executor = ThreadPoolExecutor(max_workers=4)

nlp = spacy.load("en_core_web_lg")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

reader = easyocr.Reader(['en'])

def extract_text(img_path):
    results = reader.readtext(img_path)
    extracted_text = ''
    for (bbox, text, prob) in results:
        extracted_text += text + '\n'
    return extracted_text

def extract_names(text):
    if len(text.split()) <= 2:
        text = f"Person {text} went to the store."
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def maintain_chat_structure(extracted_text):
    time_pattern = re.compile(r'(\d{1,2}[.:-]\d{2}(?:\s?[APMapm]{2})?)')
    date_pattern = re.compile(r'(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})')
    current_speaker = None
    current_date = None
    conversation = []
    lines = extracted_text.split('\n')

    for i, line in enumerate(lines):
        line = clean_text(line)
        time_match = re.search(time_pattern, line)
        date_match = re.search(date_pattern, line)

        if date_match:
            current_date = date_match.group(1)

        if time_match and i > 0:
            previous_line = clean_text(lines[i - 1]).strip()
            name = extract_names(previous_line)

            if name:
                current_speaker = name[0]
            else:
                current_speaker = "You"

            conversation.append({
                "speaker": current_speaker,
                "message": "",
                "time": time_match.group(1),
                "date": current_date
            })
        elif current_speaker:
            name_check = extract_names(line)
            if not name_check:
                if line.strip():
                    conversation[-1]["message"] += line + " "

    return conversation


def summarize_conversation(conversation):
    detailed_summary = []

    for entry in conversation:
        speaker = entry['speaker']
        time = entry['time']
        date = entry.get('date', 'Unknown date')
        message = entry['message']

        summary = summarizer(message, min_length=10, max_length=50, do_sample=False)
        summarized_text = summary[0]['summary_text']

        detailed_summary.append(f"On {date}, at {time}, {speaker} mentioned that {summarized_text}.")

    return " ".join(detailed_summary)

def process_image_and_summarize(image_path):
    extracted_text = extract_text(image_path)
    conversation = maintain_chat_structure(extracted_text)
    return summarize_conversation(conversation)


@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        
        image = request.files['image']
        image_path = image.filename
        image.save(image_path)

        future = executor.submit(process_image_and_summarize, image_path)
        summarized_content = future.result()

        return jsonify({"summary": summarized_content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
