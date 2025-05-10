import os
import json
import spacy
from PIL import Image
import pytesseract


def extract_text_from_images(folder_path):
    """Extract text from image files in the given folder using OCR."""
    image_texts = []
    supported_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in supported_ext):
            image_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img).strip()
                if text:
                    image_texts.append({"filename": filename, "text": text})
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return image_texts


def auto_annotate_and_save_json(image_texts, json_path):
    """Annotate text using a SpaCy NER model and save structured annotations."""
    try:
        nlp = spacy.load("en_core_web_trf")  # Transformer model
    except OSError:
        print("Warning: en_core_web_trf not found. Falling back to en_core_web_sm.")
        nlp = spacy.load("en_core_web_sm")  # Fallback

    annotated_data = []

    for image_entry in image_texts:
        filename, text = image_entry["filename"], image_entry["text"]
        doc = nlp(text)

        entities = []
        for ent in doc.ents:
            start = text.find(ent.text)
            if start != -1:
                entities.append({
                    "start": start,
                    "end": start + len(ent.text),
                    "label": ent.label_
                })

        if entities:
            annotated_data.append({"filename": filename, "text": text, "entities": entities})

    # Save to JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotated_data, f, indent=4, ensure_ascii=False)

    print(f"Extracted entities saved to {json_path}")


if __name__ == "__main__":
    folder_path = r"C:\Users\ACER\Desktop\pro 2\ImageFolder"
    json_path = r"C:\Users\ACER\Desktop\pro 2\Entities_from_images.json"

    image_texts = extract_text_from_images(folder_path)

    if image_texts:
        auto_annotate_and_save_json(image_texts, json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            structured_data = json.load(f)
        print(json.dumps(structured_data[:2], indent=4))  # First 2 records
    else:
        print("No valid images with text found in the provided folder.")
