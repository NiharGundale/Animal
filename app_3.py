import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import pymongo
import bcrypt
import base64 # Required for Base64 encoding for download link
from fpdf import FPDF # Required for PDF generation (fpdf2 package)

# --- Configuration & Initialization ---

# Initialize the MongoDB URI by accessing the secrets loaded by Streamlit
try:
    # Access the "uri" key within the "[mongo]" section of secrets.toml
    MONGO_URI = st.secrets["mongo"]["uri"] 
except KeyError:
    # Fallback for local development or if secrets are missing
    st.error("MongoDB URI not found in Streamlit secrets. Please check your secrets.toml file.")
    st.stop()
    
DB_NAME = "animal_detector_db" 
COLLECTION_NAME = "users" 

# Initialize Streamlit Session State for Authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# === MongoDB Functions ===
@st.cache_resource
def init_mongo_connection():
    """Initializes and caches the local MongoDB connection."""
    try:
        # Connect to the local MongoDB instance
        client = pymongo.MongoClient(MONGO_URI)
        
        # Check connection immediately
        client.admin.command('ping') 
        
        db = client[DB_NAME]
        return db[COLLECTION_NAME]
    except Exception as e:
        # This error usually means MongoDB is not running locally.
        st.error(f"❌ Failed to connect to local MongoDB at {MONGO_URI}. Is the MongoDB service running? Error: {e}")
        # Stop the app execution if the database connection fails
        st.stop()

# Initialize the users collection
try:
    users_collection = init_mongo_connection()
except SystemExit:
    # If init_mongo_connection failed and called st.stop(), we handle it here
    pass


def check_credentials(username, password):
    """Checks if a user exists and the password is correct."""
    user_data = users_collection.find_one({"username": username})
    if user_data and 'password_hash' in user_data:
        # Check password using bcrypt
        if bcrypt.checkpw(password.encode('utf-8'), user_data['password_hash']):
            return True
    return False

def add_user(username, password):
    """Adds a new user to the database."""
    if users_collection.find_one({"username": username}):
        return False, "User already exists. Please login."
    
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Insert new user
    users_collection.insert_one({
        "username": username,
        "password_hash": hashed_password
    })
    return True, "User registered successfully! You can now log in."

# === Page Configuration ===
st.set_page_config(
    page_title=" Animal Disease Detector",
    page_icon="",
    layout="wide" 
)

# === Model Loading ===
@st.cache_resource
def load_all_models():
    """Loads and caches the Keras models."""
    try:
        cow_model = load_model('cow_disease_model.h5')
        poultry_model = load_model('poultry_disease_model.keras')
        return cow_model, poultry_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}. Ensure model files are present (cow_disease_model.h5 and poultry_disease_model.keras).")
        st.stop()

# Load models only if a user is successfully logged in
cow_model, poultry_model = None, None
if st.session_state.logged_in:
    cow_model, poultry_model = load_all_models()

# === Class Labels (Hardcoded) ===
cow_class_names = ['healthycows', 'lumpycows']
poultry_class_labels = ['Bumblefoot', 'Fowlpox', 'Healthy', 'Unlabeled', 'coryza', 'crd', 'not_poultry'] 

# === Disease Info Dictionaries (Em dashes replaced with hyphens for FPDF compatibility) ===
cow_disease_info = { 
    "healthycows": {
        "description": "The cow appears healthy with no visible disease symptoms. Keep up the good work!",
        "treatment": "Continue proper nutrition, hygiene, and maintain your regular vaccination schedule."
    },
    "lumpycows": {
        # FIX: Replaced '—' (em dash) with '--' to avoid UnicodeEncodeError in latin-1
        "description": "Lumpy Skin Disease (LSD) detected--a viral infection spread by insects, causing skin nodules.",
        "treatment": "Isolate infected animals immediately, provide antiviral medication, and implement strict fly control measures. Consult a veterinarian."
    }
}
poultry_disease_info = { 
    "Healthy": {
        "description": "The bird appears healthy and shows no visible disease symptoms.",
        "treatment": "Continue providing a balanced diet, clean water, and proper coop hygiene to prevent disease."
    },
    "coryza": {
        # FIX: Replaced '—' (em dash) with '--' to avoid UnicodeEncodeError in latin-1
        "description": "Infectious Coryza--a bacterial disease causing swelling of the face, foul-smelling nasal discharge, and reduced egg production.",
        "treatment": "Treat with broad-spectrum antibiotics (like sulfonamides) and improve ventilation immediately."
    },
    "crd": {
        # FIX: Replaced '—' (em dash) with '--' to avoid UnicodeEncodeError in latin-1
        "description": "Chronic Respiratory Disease (CRD)--caused by Mycoplasma gallisepticum, leading to coughing, sneezing, and weight loss.",
        "treatment": "Administer Tylosin or Tiamulin as prescribed by a vet and disinfect housing areas thoroughly."
    },
    "Fowlpox": {
        # FIX: Replaced '—' (em dash) with '--' to avoid UnicodeEncodeError in latin-1
        "description": "Fowlpox--a slow-spreading viral disease that causes lesions (scabs) on the comb, wattles, and eyelids.",
        "treatment": "Vaccinate healthy birds in the flock and isolate infected ones. Maintain a clean and dry coop environment."
    },
    "Bumblefoot": {
        "description": "A bacterial infection (Staphylococcus) on the footpad due to injury, poor sanitation, or incorrect perching.",
        "treatment": "Clean affected area, soak the foot, apply antiseptic, and consult a vet for minor surgery and antibiotics."
    }
}
# === Image Preprocessing ===
def preprocess_image(pil_image, target_size=(224, 224)):
    """Resizes and normalizes the image for model prediction."""
    img = pil_image.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# === PDF Generation Function ===
def create_download_link(pdf_content, filename):
    """
    Generates a download link for a PDF file using Base64 encoding.
    """
    # Use output(dest='S') to get the PDF as a string buffer, then encode to Base64
    # The encoding issue is resolved by ensuring the input text (disease info)
    # does not contain characters outside the latin-1 range (like the em-dash).
    b64 = base64.b64encode(pdf_content.output(dest='S').encode('latin-1')).decode('latin-1')
    
    # Create the Markdown download link
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.pdf">⬇️ Download Analysis Report (PDF)</a>'
    
    return href

# ------------------------------------
# --- Authentication UI and Logic (CENTERED) ---
# ------------------------------------

def logout():
    """Handles user logout."""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

def login_form():
    """Renders the centered login/signup UI."""
    
    st.title("🐄 Animal Disease Detector")
    st.markdown("---")
    
    # Use columns to center the form: [1 | 1 | 1] ratio
    col_empty1, col_form, col_empty2 = st.columns([1, 1, 1])
    
    with col_form:
        st.subheader("🔒 User Authentication")
        
        auth_choice = st.radio("Select Mode", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")
        
        with st.form("auth_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button(auth_choice, use_container_width=True, type="primary")
            
            if submitted:
                if auth_choice == "Login":
                    if check_credentials(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password.")
                
                elif auth_choice == "Sign Up":
                    if len(username) < 3 or len(password) < 6:
                        st.error("Username must be at least 3 chars, password at least 6.")
                    else:
                        success, message = add_user(username, password)
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")

# ------------------------------------
# --- Main Application Logic (Protected) ---
# ------------------------------------

def main_app():
    """The core Animal Disease Detector functionality."""
    
    st.title("Animal Disease Detector")
    st.write(f"Welcome back, **{st.session_state.username}**! Use the controls to upload an image for analysis.")
    
    # Logout button in the main sidebar
    with st.sidebar:
        st.button("Logout", on_click=logout, type="primary", use_container_width=True)
        st.write("---")
    
    # Main UI Columns
    col1, col2 = st.columns([2, 3])

    # --- Column 1: Controls ---
    with col1:
        st.subheader("Controls")
        animal_type = st.selectbox("1. Select Animal Type:", ["Cow", "Poultry"])
        uploaded_file = st.file_uploader("2. Choose an image...", type=["jpg", "jpeg", "png"])
        
        analyze_button = st.button("3. Analyze Image", use_container_width=True)
        
        st.write("---") 

    # --- Column 2: Output ---
    with col2:
        st.subheader("Analysis Output")
        
        # 1. Handle image display
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            st.image(pil_img, caption="Your Uploaded Image", use_container_width=True)
        else:
            st.info("⬆️ Please upload an image using the controls on the left.")

        # 2. Handle analysis on button click
        if analyze_button:
            if uploaded_file is not None:
                with st.spinner("🔬 Analyzing image..."):
                    
                    img_array = preprocess_image(pil_img)

                    # --- Run Prediction ---
                    if animal_type == "Cow":
                        preds = cow_model.predict(img_array)
                        prediction = preds[0][0]
                        
                        # Assuming a binary model: 0 is healthy, 1 is lumpy
                        class_index = int(prediction > 0.5) 
                        confidence = prediction if class_index == 1 else 1 - prediction
                        predicted_label = cow_class_names[class_index]
                        info = cow_disease_info.get(predicted_label, {})
                        model_used = "Cow Disease Model"

                    elif animal_type == "Poultry":
                        preds = poultry_model.predict(img_array)
                        pred_idx = np.argmax(preds[0])
                        confidence = np.max(preds[0])
                        predicted_label = poultry_class_labels[pred_idx]
                        info = poultry_disease_info.get(predicted_label, {})
                        model_used = "Poultry Disease Model"
                        
                    
                    # --- Collect final results ---
                    diagnosis_result = predicted_label.upper()
                    confidence_percent = f"{confidence*100:.2f}%"
                    description = info.get('description', 'No detailed information available.')
                    treatment = info.get('treatment', 'No treatment information available. Consult a veterinarian immediately.')

                    # Handle 'not_poultry' or 'Unlabeled'
                    if predicted_label in ['Unlabeled', 'not_poultry']:
                        st.warning(f"**Diagnosis:** ⚠️ {diagnosis_result} ({confidence_percent}) - The model suggests this image may not be suitable for analysis or falls into an unknown category.")
                        st.info("Please try another image that clearly shows the animal.")
                    else:
                        # --- Display Results ---
                        st.success(f"**Diagnosis:** ✅ **{diagnosis_result}** ({confidence_percent})")
                        st.markdown("---")
                        st.info(f"**Description:**\n{description}")
                        st.warning(f"**Recommended Action:**\n{treatment}")
                        
                        # =========================================================
                        # === PDF GENERATION & DOWNLOAD LOGIC ===
                        # =========================================================
                        
                        # 1. Create PDF object
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        
                        # 2. Add content to PDF
                        pdf.cell(200, 10, txt="Animal Disease Detection Report", ln=1, align="C")
                        pdf.ln(5)

                        # Write metadata
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt=f"Analysis for: {animal_type}", ln=1)
                        pdf.cell(200, 10, txt=f"Model Used: {model_used}", ln=1)
                        pdf.ln(5)

                        # Write Diagnosis
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(200, 10, txt=f"DIAGNOSIS: {diagnosis_result}", ln=1)
                        pdf.set_font("Arial", '', 12)
                        pdf.cell(200, 10, txt=f"Confidence: {confidence_percent}", ln=1)
                        
                        pdf.ln(5)
                        
                        # Write Description
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Description:", ln=1)
                        pdf.set_font("Arial", '', 12)
                        # The multi_cell function handles text wrapping
                        pdf.multi_cell(0, 10, txt=description)
                        
                        pdf.ln(5)

                        # Write Treatment
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Recommended Action:", ln=1)
                        pdf.set_font("Arial", '', 12)
                        pdf.multi_cell(0, 10, txt=treatment)


                        # 3. Create the download link in Streamlit
                        filename = f"{animal_type}_{predicted_label.replace(' ', '_')}_report"
                        download_link = create_download_link(pdf, filename)
                        
                        st.markdown("---")
                        st.markdown(download_link, unsafe_allow_html=True)
                        st.markdown("---")

                        # =========================================================
            
            else:
                st.error("🛑 Please upload an image first before analyzing.")

# ------------------------------------
# --- Master Control Flow ---
# ------------------------------------
if st.session_state.logged_in:
    main_app()
else:
    login_form()