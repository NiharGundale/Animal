import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# === Page Configuration ===
# Using "wide" layout to give our columns more space
st.set_page_config(
    page_title=" Animal Disease Detector",
    page_icon="🐄",
    layout="wide" 
)

# === Model Loading ===
@st.cache_resource
def load_all_models():
    try:
        cow_model = load_model('cow_disease_model.h5')
        poultry_model = load_model('poultry_disease_model.keras')
        return cow_model, poultry_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

cow_model, poultry_model = load_all_models()

# === Class Labels (Hardcoded) ===
cow_class_names = ['healthycows', 'lumpycows']
poultry_class_labels = ['Bumblefoot', 'Fowlpox', 'Healthy', 'Unlabeled', 'coryza', 'crd', 'not_poultry'] # <-- VERIFY THIS ORDER!

# === Disease Info Dictionaries ===
# (No changes to your dictionaries)
cow_disease_info = {  "healthycows": {
        "description": "The cow appears healthy with no visible disease symptoms.",
        "treatment": "Continue proper nutrition, hygiene, and vaccination schedule."
    },
    "lumpycows": {
        "description": "Lumpy Skin Disease — a viral infection spread by insects.",
        "treatment": "Isolate infected animals, provide antiviral medication, and maintain fly control. Consult a veterinarian immediately."
    }
}
poultry_disease_info = { "Healthy": {
        "description": "The bird appears healthy and shows no disease symptoms.",
        "treatment": "Continue providing a balanced diet, clean water, and proper coop hygiene."
    },
    "coryza": {
        "description": "Infectious Coryza — a bacterial disease causing swelling of the face and nasal discharge.",
        "treatment": "Treat with broad-spectrum antibiotics and maintain proper ventilation."
    },
    "crd": {
        "description": "Chronic Respiratory Disease — caused by Mycoplasma gallisepticum infection.",
        "treatment": "Administer Tylosin or Tiamulin as prescribed and disinfect housing areas."
    },
    "Fowlpox": {
        "description": "Fowlpox — a viral disease that causes lesions on the comb, wattles, and eyelids.",
        "treatment": "Vaccinate healthy birds and isolate infected ones. Keep coop clean and dry."
    },
    "Bumblefoot": {
        "description": "A bacterial infection on the footpad due to injury or poor sanitation.",
        "treatment": "Clean affected area, apply antiseptic, and consult a vet for antibiotics."
    }
}
# === Image Preprocessing ===
def preprocess_image(pil_image, target_size=(224, 224)):
    img = pil_image.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# === Streamlit App Interface ===
st.title(" Animal Disease Detector")
st.write("Upload an image for analysis. Controls are on the left, results are on the right.")

# --- START OF UI CHANGES ---

# Create two columns. 
# We'll give the left column 2 parts and the right 3 parts (more space for image/results)
col1, col2 = st.columns([2, 3])

# --- Column 1: Controls ---
with col1:
    st.subheader("Controls")
    animal_type = st.selectbox("1. Select Animal Type:", ["Cow", "Poultry"])
    uploaded_file = st.file_uploader("2. Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Add the "Submit" button here
    analyze_button = st.button("3. Analyze Image")
    
    # Add a placeholder to show the button is in the column
    st.write("---") 

# --- Column 2: Output ---
with col2:
    st.subheader("Analysis Output")
    
    # 1. Handle image display
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Your Uploaded Image", use_container_width=True)
    else:
        st.info("Please upload an image using the controls on the left.")

    # 2. Handle analysis on button click
    if analyze_button:
        if uploaded_file is not None:
            # Only run if the button is clicked AND there is a file
            with st.spinner("Analyzing image..."):
                
                # Preprocess image (using the pil_img from above)
                img_array = preprocess_image(pil_img)

                # --- Run Prediction ---
                if animal_type == "Cow":
                    preds = cow_model.predict(img_array)
                    prediction = preds[0][0]
                    class_index = int(prediction > 0.5)
                    confidence = prediction if class_index == 1 else 1 - prediction
                    predicted_label = cow_class_names[class_index]
                    info = cow_disease_info.get(predicted_label, {})

                elif animal_type == "Poultry":
                    preds = poultry_model.predict(img_array)
                    pred_idx = np.argmax(preds[0])
                    confidence = np.max(preds[0])
                    predicted_label = poultry_class_labels[pred_idx]
                    info = poultry_disease_info.get(predicted_label, {})

                # --- Display Results ---
                st.success(f"**Diagnosis:** {predicted_label} ({confidence*100:.2f}%)")
                st.info(f"**Description:**\n{info.get('description', 'No info available.')}")
                st.warning(f"**Recommended Action:**\n{info.get('treatment', 'No treatment information available.')}")
        
        else:
            # Show error if button clicked with no file
            st.error("Please upload an image first before analyzing.")

# --- END OF UI CHANGES ---