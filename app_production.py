import os
import gdown
from flask import Flask, render_template, request, flash, send_from_directory, redirect, url_for
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.contrib.github import make_github_blueprint, github
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from keras.models import load_model
import tensorflow as tf
import numpy as np
import json
import uuid
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'gautamtyagi7906')

# ------------------ Database + Auth Config ------------------ #
# Production database configuration
if os.environ.get('DATABASE_URL'):
    database_url = os.environ.get('DATABASE_URL')
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Login manager setup
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

# ------------------ Flask Mail Config ------------------ #
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', "gautamtyagi9058@gmail.com")
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', "qsqmbojpbubkbtjn")
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME', "gautamtyagi9058@gmail.com")

mail = Mail(app)

# Google OAuth Setup
google_bp = make_google_blueprint(
    client_id=os.environ.get('GOOGLE_CLIENT_ID', "6209381610-6p0s9vf8ecei0om42lmp83ff4vvgp8u9.apps.googleusercontent.com"),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', "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"),
    scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email",
    ],
    redirect_to="google_login" 
)
app.register_blueprint(google_bp, url_prefix="/login")

# GitHub OAuth Setup
github_bp = make_github_blueprint(
    client_id=os.environ.get('GITHUB_CLIENT_ID', "Ov23liVo5vNdwneQvWCE"),
    client_secret=os.environ.get('GITHUB_CLIENT_SECRET', "668c1855d2141f967c61a57fff36fdca9f56e536"),
)
app.register_blueprint(github_bp, url_prefix="/login")

# ------------------ Model Download and Load Functions ------------------ #
def download_model_from_drive(file_id, output_path):
    """Download model from Google Drive using file ID"""
    try:
        print(f"Downloading model to {output_path}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"Model downloaded successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def ensure_models_exist():
    """Check if models exist, if not download them from Google Drive"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model1_path = os.path.join(models_dir, "crop_disease_detection.keras")
    model2_path = os.path.join(models_dir, "crop_disease_detection1.keras")
    
    # Google Drive file IDs for your models (you need to replace these with actual IDs)
    # To get file ID: Share your file on Google Drive → Get shareable link → Extract ID from URL
    MODEL1_DRIVE_ID = os.environ.get('MODEL1_DRIVE_ID', 'YOUR_MODEL1_FILE_ID_HERE')
    MODEL2_DRIVE_ID = os.environ.get('MODEL2_DRIVE_ID', 'YOUR_MODEL2_FILE_ID_HERE')
    
    # Download model1 if not exists
    if not os.path.exists(model1_path):
        print("Model 1 not found. Downloading from Google Drive...")
        if not download_model_from_drive(MODEL1_DRIVE_ID, model1_path):
            print("Failed to download Model 1")
            return False, False
    
    # Download model2 if not exists
    if not os.path.exists(model2_path):
        print("Model 2 not found. Downloading from Google Drive...")
        if not download_model_from_drive(MODEL2_DRIVE_ID, model2_path):
            print("Failed to download Model 2")
            return False, False
    
    return os.path.exists(model1_path), os.path.exists(model2_path)

def load_models():
    """Load ML models with automatic download if needed"""
    try:
        # Ensure models exist (download if needed)
        model1_exists, model2_exists = ensure_models_exist()
        
        if not model1_exists or not model2_exists:
            print("Models could not be downloaded. Using fallback mode.")
            return None, None, []
        
        # Load models
        model1 = tf.keras.models.load_model("models/crop_disease_detection.keras")
        model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        model2 = tf.keras.models.load_model("models/crop_disease_detection1.keras")
        model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Load plant disease data
        with open("plant_disease.json") as f:
            plant_disease = json.load(f)
        
        print("✅ Models loaded successfully!")
        return model1, model2, plant_disease
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, []

# ------------------ Models Load ------------------ #
print("Initializing models...")
model1, model2, plant_disease = load_models()

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ------------------ File Upload Config ------------------ #
UPLOAD_FOLDER = 'static/uploadimages'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/uploadimages/<filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ------------------ OAuth Routes ------------------ #
@app.route("/google_login")
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))

    resp = google.get("https://www.googleapis.com/oauth2/v2/userinfo")
    if resp.ok:
        user_info = resp.json()
        email = user_info["email"]
        username = user_info.get("name", email.split("@")[0])

        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(username=username, email=email, password=bcrypt.generate_password_hash("oauth_dummy").decode('utf-8'))
            db.session.add(user)
            db.session.commit()

        login_user(user)
        flash("✅ Logged in with Google!", "success")
        return redirect(url_for("home"))
    flash("❌ Google login failed!", "danger")
    return redirect(url_for("login"))

@app.route("/github_login")
def github_login():
    if not github.authorized:
        return redirect(url_for("github.login"))

    resp = github.get("/user")
    if resp.ok:
        user_info = resp.json()
        email = user_info.get("email") or f"{user_info['login']}@github.com"
        username = user_info["login"]

        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(username=username, email=email, password=bcrypt.generate_password_hash("oauth_dummy").decode('utf-8'))
            db.session.add(user)
            db.session.commit()

        login_user(user)
        flash("✅ Logged in with GitHub!", "success")
        return redirect(url_for("home"))
    flash("❌ GitHub login failed!", "danger")
    return redirect(url_for("login"))

# ------------------ Auth Routes ------------------ #
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/welcome")
def welcome():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    user_info = resp.json()
    return f"Hello, {user_info['email']}!"

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")

        if password != confirm:
            flash("Passwords do not match!", "error")
            return redirect(url_for("signup"))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("❌ This email is already registered. Please try another email!", "danger")
            return redirect(url_for("signup"))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()

        flash("✅ Account Created successfully! Now login", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            flash("✅ Login Successful! Welcome to AgriCare Dashboard।", "success")
            return redirect(next_page) if next_page else redirect(url_for("home"))
        else:
            flash("❌ Wrong Email or Password! Please try again।", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route('/create_db')
def create_db():
    db.create_all()
    return "Database created successfully!"

# ------------------ Pages ------------------ #
@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")

        if not name or not email or not message:
            flash("⚠️ Please fill all fields!", "danger")
        else:
            try:
                msg = Message(subject=f"New Contact Form: {subject}", recipients=["anshuktyagi589@gmail.com"])
                msg.body = f"""
                👤 Name: {name}
                📧 Email: {email}
                📝 Subject: {subject}
                💬 Message: {message}
                """
                mail.send(msg)
                flash("✅ Message sent successfully!", "success")
            except Exception as e:
                print("❌ Error sending email:", e)
                flash("❌ Failed to send message.", "danger")
    return render_template("contact.html")

@app.route('/faq')
def faqs():
    return render_template('faq.html')

@app.route('/cookies')
def cookies():
    return render_template('cookies.html')

@app.route('/studies')
def studies():
    return render_template('studies.html')

@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

@app.route('/whitepapers')
def whitepapers():
    return render_template('whitepapers.html')

# ------------------ Prediction Functions ------------------ #
def extract_features(image_path):
    if not model1 or not model2:
        return None
    try:
        image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
        feature = tf.keras.utils.img_to_array(image)
        feature = np.array([feature])
        return feature
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def get_disease_info(class_name):
    for disease in plant_disease:
        if disease['name'] == class_name:
            return disease.copy()
    return {'name': class_name, 'cause': 'Info not available', 'cure': 'Consult expert'}

def ensemble_predict(image_path):
    if not model1 or not model2:
        return {
            'name': 'Model Error', 
            'cause': 'Models are not loaded. Please try again later.', 
            'cure': 'Contact support if problem persists', 
            'confidence': 0,
            'confidence_int': 0,
            'model1_prediction': 'N/A',
            'model2_prediction': 'N/A',
            'model1_confidence': 0,
            'model2_confidence': 0
        }
        
    img = extract_features(image_path)
    if img is None:
        return {
            'name': 'Processing Error', 
            'cause': 'Image processing failed', 
            'cure': 'Please try uploading a different image', 
            'confidence': 0,
            'confidence_int': 0,
            'model1_prediction': 'N/A',
            'model2_prediction': 'N/A',
            'model1_confidence': 0,
            'model2_confidence': 0
        }
        
    try:
        prediction1 = model1.predict(img, verbose=0)
        prediction2 = model2.predict(img, verbose=0)

        model1_index = np.argmax(prediction1)
        model2_index = np.argmax(prediction2)

        model1_class = class_names[model1_index]
        model2_class = class_names[model2_index]

        model1_conf = round(float(np.max(prediction1)) * 100, 2)
        model2_conf = round(float(np.max(prediction2)) * 100, 2)

        if model2_conf >= 80 and model2_conf > model1_conf:
            final_prediction = model2_class
            final_confidence = model2_conf
            used_model = "Model 2"
        elif model1_conf >= 80 and model1_conf > model2_conf:
            final_prediction = model1_class
            final_confidence = model1_conf
            used_model = "Model 1"
        else:
            ensemble_prediction = (prediction1 + prediction2) / 2
            ensemble_index = np.argmax(ensemble_prediction)
            final_prediction = class_names[ensemble_index]
            final_confidence = round(float(np.max(ensemble_prediction)) * 100, 2)
            used_model = "Ensemble"

        prediction_info = get_disease_info(final_prediction)
        prediction_info['confidence'] = final_confidence
        prediction_info['confidence_int'] = int(final_confidence)
        prediction_info['model1_prediction'] = model1_class
        prediction_info['model2_prediction'] = model2_class
        prediction_info['model1_confidence'] = model1_conf
        prediction_info['model2_confidence'] = model2_conf
        prediction_info['used_model'] = used_model
        
        return prediction_info
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {
            'name': 'Prediction Error', 
            'cause': 'Error occurred during prediction', 
            'cure': 'Please try again with a different image', 
            'confidence': 0,
            'confidence_int': 0,
            'model1_prediction': 'N/A',
            'model2_prediction': 'N/A',
            'model1_confidence': 0,
            'model2_confidence': 0
        }

# ------------------ Upload Handler ------------------ #
@app.route('/upload/', methods=['POST'])
def uploadimage():
    try:
        if 'img' not in request.files:
            flash("❌ No image file selected!", "danger")
            return redirect(url_for("home"))
            
        image_file = request.files['img']
        if image_file.filename == '':
            flash("❌ No image file selected!", "danger")
            return redirect(url_for("home"))
            
        filename = f"temp_{uuid.uuid4().hex}_{image_file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(filepath)

        prediction = ensemble_predict(filepath)
        image_path_for_template = f"uploadimages/{filename}"

        return render_template('result.html', prediction=prediction, imagepath=image_path_for_template)
        
    except Exception as e:
        print(f"Error in upload handler: {e}")
        flash("❌ Error processing image. Please try again.", "danger")
        return redirect(url_for("home"))

# Health check route for deployment
@app.route('/health')
def health_check():
    return {
        'status': 'healthy', 
        'models_loaded': model1 is not None and model2 is not None,
        'model1_status': 'loaded' if model1 else 'not loaded',
        'model2_status': 'loaded' if model2 else 'not loaded'
    }

# Route to manually reload models (for debugging)
@app.route('/reload_models')
def reload_models():
    global model1, model2, plant_disease
    try:
        model1, model2, plant_disease = load_models()
        if model1 and model2:
            return "✅ Models reloaded successfully!"
        else:
            return "❌ Failed to reload models"
    except Exception as e:
        return f"❌ Error reloading models: {e}"

# ------------------ Run ------------------ #
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)