# models.py - Neural network loaders and predictors
from PIL import Image
from config import DEFAULT_MODEL

# Safe Streamlit caching decorator fallback for non-streamlit testing
try:
    import streamlit as st
    cache_decorator = st.cache_resource
except Exception:
    def cache_decorator(func):
        return func

@cache_decorator
def get_classifier_pipeline(model_id: str = DEFAULT_MODEL):
    """
    Loads and caches the Hugging Face image classification pipeline.
    """
    from transformers import pipeline
    # The pipeline will load weights from cache or download on demand.
    # It automatically selects CPU/GPU based on availability.
    return pipeline("image-classification", model=model_id)

def classify_image(image: Image.Image, model_id: str = DEFAULT_MODEL) -> dict:
    """
    Predicts whether an image is REAL or FAKE (AI-generated)
    using the fine-tuned Hugging Face transformer model.
    """
    # Convert image to RGB mode if not already
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    try:
        classifier = get_classifier_pipeline(model_id)
        raw_predictions = classifier(image)
        
        # Standardize labels from the classifier
        # dima806/ai_vs_real_image_detection returns [{'label': 'FAKE', ...}, {'label': 'REAL', ...}]
        predictions = {}
        for pred in raw_predictions:
            label = pred["label"].upper()
            predictions[label] = float(pred["score"])
            
        # We retrieve scores by checking both ViT ("FAKE"/"REAL") and SigLIP ("AI"/"HUM" or "HUMAN") labels
        fake_score = predictions.get("FAKE", predictions.get("AI", predictions.get("SYNTHETIC", 0.0)))
        real_score = predictions.get("REAL", predictions.get("HUMAN", predictions.get("HUM", predictions.get("ORIGINAL", 0.0))))
        
        # If neither is found or one is missing, fall back to sub-string matching
        if fake_score == 0.0 and real_score == 0.0:
            for label, score in predictions.items():
                if any(x in label for x in ["FAKE", "AI", "SYNTHETIC"]):
                    fake_score = score
                elif any(x in label for x in ["REAL", "HUMAN", "HUM", "ORIGINAL"]):
                    real_score = score
                    
        # Guarantee they sum to 1.0
        total = fake_score + real_score
        if total > 0:
            fake_score /= total
            real_score /= total
        else:
            fake_score, real_score = 0.5, 0.5
            
        verdict = "AI" if fake_score >= 0.50 else "REAL"
        
        return {
            "success": True,
            "fake_score": fake_score,
            "real_score": real_score,
            "verdict": verdict,
            "raw": raw_predictions
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fake_score": 0.5,
            "real_score": 0.5,
            "verdict": "ERROR"
        }

@cache_decorator
def get_category_pipeline():
    """
    Loads a highly optimized, lightweight MobileNet model (~13MB) 
    for fast category classification of real images.
    """
    from transformers import pipeline
    return pipeline("image-classification", model="google/mobilenet_v2_1.0_224")

def classify_real_category(image: Image.Image) -> str:
    """
    Sub-classifies a REAL image into categories: Human, Animal, Bird, Insect, or Object/Thing.
    """
    try:
        classifier = get_category_pipeline()
        raw_preds = classifier(image)
        if not raw_preds:
            return "Object/Thing"
            
        top_pred = raw_preds[0]
        label = top_pred["label"].lower()
        
        # Mapping keyword tags
        human_kws = ["face", "person", "human", "woman", "man", "head", "portrait", "child", "groom", "bride", "military", "doctor", "tshirt", "jeans", "suit", "cloak", "wig"]
        animal_kws = ["dog", "cat", "bear", "lion", "tiger", "wolf", "fox", "horse", "cow", "pig", "sheep", "elephant", "monkey", "ape", "snake", "lizard", "frog", "turtle", "mammal", "rodent", "mouse", "rat", "deer", "koala", "panda", "rabbit", "squirrel"]
        bird_kws = ["bird", "eagle", "hawk", "owl", "parrot", "penguin", "sparrow", "swan", "duck", "goose", "chicken", "macaw", "finch", "flamingo", "peacock", "ostrich", "vulture", "cockatoo"]
        insect_kws = ["insect", "spider", "ant", "bee", "wasp", "butterfly", "moth", "beetle", "fly", "grasshopper", "cricket", "dragonfly", "caterpillar", "scorpion", "spiderweb", "tick"]
        
        if any(kw in label for kw in human_kws):
            return "Human"
        elif any(kw in label for kw in bird_kws):
            return "Bird"
        elif any(kw in label for kw in animal_kws):
            return "Animal"
        elif any(kw in label for kw in insect_kws):
            return "Insect"
        else:
            # Fallback checking of secondary prediction entries
            for pred in raw_preds[1:3]:
                lbl = pred["label"].lower()
                if any(kw in lbl for kw in human_kws):
                    return "Human"
                elif any(kw in lbl for kw in bird_kws):
                    return "Bird"
                elif any(kw in lbl for kw in animal_kws):
                    return "Animal"
                elif any(kw in lbl for kw in insect_kws):
                    return "Insect"
                    
            return "Object/Thing"
    except Exception as e:
        print(f"Real Image sub-classification failed: {e}")
        return "Object/Thing"
