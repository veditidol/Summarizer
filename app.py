from flask import Flask, request, jsonify
import easyocr
import re
import spacy
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load spaCy model and summarization model
nlp = spacy.load("en_core_web_lg")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
reader = easyocr.Reader(['en'])

# Function to extract text data from images
def extract_text(img_path):
    results = reader.readtext(img_path)
    extracted_text = ''
    for (bbox, text, prob) in results:
        extracted_text += text + '\n'
    return extracted_text

# Function to extract names from text
def extract_names(text):
    if len(text.split()) <= 2:
        text = f"Person {text} went to the store."
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

# Function to clean the extracted text
def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

# Function to maintain the chat structure
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

# Function to summarize the conversation
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

# Flask route to process the image
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Save the uploaded image
        image = request.files['image']
        image_path = image.filename
        image.save(image_path)

        # Extract text from the image
        extracted_text = extract_text(image_path)

        # Maintain chat structure
        conversation = maintain_chat_structure(extracted_text)

        # Summarize the conversation
        summarized_content = summarize_conversation(conversation)

        # Return the summarized conversation as JSON
        return jsonify({"summary": summarized_content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

