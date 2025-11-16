import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import pymongo
import bcrypt

# --- Configuration & Initialization ---

# 🚨 IMPORTANT: URI for a local MongoDB installation.
# Ensure your local MongoDB service (mongod) is running!
# === In your app.py file ===

# Initialize the MongoDB URI by accessing the secrets loaded by Streamlit
try:
    # Access the "uri" key within the "[mongo]" section of secrets.toml
    MONGO_URI = st.secrets["mongo"]["uri"] 
except KeyError:
    # Fallback for local development or if secrets are missing
    # WARNING: This fallback should not contain the real password if committed!
    st.error("MongoDB URI not found in Streamlit secrets.")
    st.stop()
    
DB_NAME = "animal_detector_db" 
COLLECTION_NAME = "users" 

# ... (rest of your MongoDB connection functions)
# MONGO_URI = "mongodb://localhost:27017/" 
# DB_NAME = "animal_detector_db" # The database where users will be stored
# COLLECTION_NAME = "users" # The collection storing user credentials

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

# Initialize the users collection (only if not running the auth form)
try:
    users_collection = init_mongo_connection()
except SystemExit:
    # If init_mongo_connection failed and called st.stop(), we need to handle it here
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
    # 1. Generate a salt (random value)
    # 2. Hash the password + salt
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
        # Load models based on their file extensions
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

# === Disease Info Dictionaries ===
cow_disease_info = { 
    "healthycows": {
        "description": "The cow appears healthy with no visible disease symptoms.",
        "treatment": "Continue proper nutrition, hygiene, and vaccination schedule."
    },
    "lumpycows": {
        "description": "Lumpy Skin Disease — a viral infection spread by insects.",
        "treatment": "Isolate infected animals, provide antiviral medication, and maintain fly control. Consult a veterinarian immediately."
    }
}
poultry_disease_info = { 
    "Healthy": {
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
    """Resizes and normalizes the image for model prediction."""
    img = pil_image.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# ------------------------------------
# --- Authentication UI and Logic ---
# ------------------------------------

def logout():
    """Handles user logout."""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun() # Use st.rerun()

def login_form():
    """Renders the login/signup UI in the sidebar."""
    st.sidebar.title("🔒 User Authentication")
    
    auth_choice = st.sidebar.radio("Select Mode", ["Login", "Sign Up"])
    
    with st.sidebar.form("auth_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(auth_choice)
        
        if submitted:
            if auth_choice == "Login":
                if check_credentials(username, password):
                    if check_credentials(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("Login successful!")
                        st.rerun() # Use st.rerun()
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
    st.sidebar.button("Logout", on_click=logout, type="primary")
    st.sidebar.write("---")
    
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
                        # This logic converts a single sigmoid output to a class index and confidence
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
                        
                    # Handle 'not_poultry' or 'Unlabeled' if they appear
                    if predicted_label in ['Unlabeled', 'not_poultry']:
                        st.warning(f"**Diagnosis:** ⚠️ {predicted_label} ({confidence*100:.2f}%) - The model suggests this image may not be suitable for analysis or falls into an unknown category.")
                        st.info("Please try another image that clearly shows the animal.")
                    else:
                        # --- Display Results ---
                        st.success(f"**Diagnosis:** ✅ **{predicted_label.upper()}** ({confidence*100:.2f}%)")
                        
                        st.markdown("---")
                        
                        st.info(f"**Description:**\n{info.get('description', 'No detailed information available.')}")
                        
                        st.warning(f"**Recommended Action:**\n{info.get('treatment', 'No treatment information available. Consult a veterinarian immediately.')}")
            
            else:
                st.error("🛑 Please upload an image first before analyzing.")

# ------------------------------------
# --- Master Control Flow ---
# ------------------------------------
if st.session_state.logged_in:
    main_app()
else:
    login_form()