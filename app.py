import os
from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from difflib import SequenceMatcher
import nltk
import ssl
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx
import pdfplumber
import traceback # For better error debugging

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure upload directory exists

# Pandas Display Options (for debugging, not strictly necessary for final output)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# --- NLTK and SpaCy Downloads (handled in app context) ---
# Disable SSL verification (temporary fix) - Ensure this block is run
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass  # For Python < 3.7
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download resources (will only download if not already present)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))


# --- Global Variables for Loaded Data and Model (Load once at startup) ---
jobs_df = None
resumes_df = None # Still needed for model training, even if not for direct upload
tfidf_vectorizer = None
job_tfidf = None
model = None

def load_and_prepare_data():
    """Loads datasets, preprocesses text, and trains the ML model."""
    global jobs_df, resumes_df, tfidf_vectorizer, job_tfidf, model

    try:
        jobs_df = pd.read_csv("jobs_dataset.csv")
        resumes_df = pd.read_csv("resumes_dataset.csv")
    except FileNotFoundError:
        print("Error: Make sure 'jobs_dataset.csv' and 'resumes_dataset.csv' are in the same directory as the script.")
        print("Or update the file paths in the code.")
        return False # Indicate failure to load data

    # Assign IDs
    jobs_df['job_id'] = jobs_df.index
    resumes_df['resume_id'] = resumes_df.index # Keep for consistency with original training

    # Text Preprocessing
    jobs_df['Job Title'] = jobs_df['Job Title'].fillna('')
    jobs_df['Job Description'] = jobs_df['Job Description'].fillna('')
    jobs_df['skills'] = jobs_df['skills'].fillna('')
    resumes_df['Resume_str'] = resumes_df['Resume_str'].fillna('') # Still needed for model training

    jobs_df['title_clean'] = jobs_df['Job Title'].apply(preprocess_text)
    jobs_df['description_clean'] = jobs_df['Job Description'].apply(preprocess_text)
    jobs_df['skills_clean'] = jobs_df['skills'].apply(preprocess_text)
    resumes_df['resume_text_clean'] = resumes_df['Resume_str'].apply(preprocess_text)

    # Create concatenated job text for vectorization
    jobs_df['job_text'] = jobs_df['title_clean'] + ' ' + jobs_df['description_clean'] + ' ' + jobs_df['skills_clean']

    # Vectorize job documents
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    job_tfidf = tfidf_vectorizer.fit_transform(jobs_df['job_text'])

    # Vectorize resumes (important: transform, not fit_transform) - for model training
    resume_tfidf_train = tfidf_vectorizer.transform(resumes_df['resume_text_clean'])

    # --- Generating 'features_df' and 'label' for ML model training ---
    features = []
    for i in range(len(resumes_df)):
        for j in range(len(jobs_df)):
            resume_vec = resume_tfidf_train[i]
            job_vec = job_tfidf[j]
            features.append({
                'resume_id': resumes_df.iloc[i]['resume_id'],
                'job_id': jobs_df.iloc[j]['job_id'],
                'cosine_sim': cosine_similarity(resume_vec, job_vec)[0][0],
                'skill_overlap': skill_overlap(resumes_df.iloc[i]['resume_text_clean'], jobs_df.iloc[j]['skills_clean']),
                'exp_match': experience_match(resumes_df.iloc[i]['resume_text_clean'], jobs_df.iloc[j]['description_clean']),
                'title_sim': title_similarity(resumes_df.iloc[i]['resume_text_clean'], jobs_df.iloc[j]['title_clean'])
            })
    features_df = pd.DataFrame(features)

    # Create the 'label' column for training
    features_df['label'] = ((features_df['cosine_sim'] > 0.05) &(features_df['skill_overlap'] > 8) & ((features_df['exp_match'] == 1) | (features_df['title_sim'] > 0.3))).astype(int)

    # --- Machine Learning Model Training ---
    X = features_df[['cosine_sim', 'skill_overlap', 'exp_match', 'title_sim']]
    y = features_df['label']

    # Split data for training the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Training the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Data loaded and ML model trained successfully!")
    return True # Indicate success

# --- Helper Functions  ---
def clean_text(text):
    """Strips HTML, removes non-alphanumeric characters, and converts to lowercase."""
    text = re.sub(r'<[^>]+>', ' ', text)  # strip HTML
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)  # non-alphanumeric chars
    return text.lower()

def preprocess_text(text):
    """Cleans, tokenizes, lemmatizes, and removes stopwords from text."""
    text = clean_text(text)
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if t.text not in stop_words and not t.is_space]
    return " ".join(tokens)

def skill_overlap(resume_text, job_skills):
    resume_tokens = set(resume_text.lower().split())
    skill_tokens = set(job_skills.lower().split())
    return len(resume_tokens.intersection(skill_tokens))

def experience_match(resume_text, job_desc):
    resume_exp_matches = re.findall(r"(\d+)\s+year", resume_text.lower())
    job_exp_matches = re.findall(r"(\d+)\s+year", job_desc.lower())
    resume_exp = int(resume_exp_matches[0]) if resume_exp_matches else 0
    job_exp = int(job_exp_matches[0]) if job_exp_matches else 0
    return 1 if resume_exp >= job_exp else 0

def title_similarity(resume_text, job_title):
    return SequenceMatcher(None, resume_text.lower(), job_title.lower()).ratio()

# Defining IR Retrieval Function
def retrieve_top_n_jobs(resume_vector, job_matrix, job_metadata, top_n=5):
    cosine_similarities = cosine_similarity(resume_vector, job_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'Job ID': job_metadata.iloc[idx]['job_id'],
            'Job Title': job_metadata.iloc[idx]['Job Title'],
            'Similarity Score': round(cosine_similarities[idx], 3),
            'Snippet': job_metadata.iloc[idx]['Job Description'][:150]
            })
    return pd.DataFrame(results)

# Hybrid (IR then ML) Recommendation Function
def get_hybrid_recommendations(uploaded_resume_text, ir_top_k=20, ml_top_k=5):
    if jobs_df is None or tfidf_vectorizer is None or job_tfidf is None or model is None:
        print("Error: Data or model not loaded. Please restart the application.")
        return pd.DataFrame()

    # Preprocess the uploaded resume text
    processed_resume_text = preprocess_text(uploaded_resume_text)
    if not processed_resume_text.strip():
        print("Warning: Uploaded resume text is empty after preprocessing.")
        return pd.DataFrame()

    # Vectorize the uploaded resume
    # We need to reshape for single sample transformation
    uploaded_resume_tfidf_vec = tfidf_vectorizer.transform([processed_resume_text])

    # Step 1: Retrieve top N jobs based on IR (e.g., top 20)
    ir_results_df = retrieve_top_n_jobs(uploaded_resume_tfidf_vec, job_tfidf, jobs_df, top_n=ir_top_k)

    if ir_results_df.empty:
        print(f"No IR-based jobs found for the uploaded resume.")
        return pd.DataFrame()

    # Get the job_ids of the top IR results
    top_ir_job_ids = ir_results_df['Job ID'].tolist()

    # Step 2: Calculate features for this resume against only the top IR jobs
    on_demand_features_list = []
    for job_id in top_ir_job_ids:
        # Find the original index of the job_id in jobs_df
        # Using .loc for faster lookup if job_id is directly the index, otherwise jobs_df[jobs_df['job_id'] == job_id].index[0]
        job_original_index = jobs_df[jobs_df['job_id'] == job_id].index[0]
        job_data = jobs_df.loc[job_original_index]
        job_tfidf_vec = job_tfidf[job_original_index] # TF-IDF vector for this job

        on_demand_features_list.append({
            'job_id': job_data['job_id'], # No resume_id here, as it's for a single resume
            'cosine_sim': cosine_similarity(uploaded_resume_tfidf_vec, job_tfidf_vec)[0][0],
            'skill_overlap': skill_overlap(processed_resume_text, job_data['skills_clean']),
            'exp_match': experience_match(processed_resume_text, job_data['description_clean']),
            'title_sim': title_similarity(processed_resume_text, job_data['title_clean'])
        })
    # Convert to DataFrame for prediction
    on_demand_features_df = pd.DataFrame(on_demand_features_list)

    # Step 3: Predict scores using the trained ML model on this subset
    X_predict = on_demand_features_df[['cosine_sim', 'skill_overlap', 'exp_match', 'title_sim']]
    on_demand_features_df['predicted_score'] = model.predict_proba(X_predict)[:, 1]

    # Step 4: Sort by predicted_score and get top N (ML_top_k) from this refined set
    top_matches = on_demand_features_df.sort_values(by='predicted_score', ascending=False).head(ml_top_k)

    # Step 5: Merge with jobs_df to get original job details
    recommended_jobs_details = pd.merge(
        top_matches,
        jobs_df[['job_id', 'Job Title', 'Job Description', 'skills']],
        on='job_id',
        how='left'
    )

    return recommended_jobs_details


# --- File Upload and Text Extraction Helpers ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() if page.extract_text() else ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    return text

def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""
    return text

def extract_text_from_file(file_path):
    file_extension = file_path.rsplit('.', 1)[1].lower()
    if file_extension == 'txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif file_extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == 'docx':
        return extract_text_from_docx(file_path)
    return ""


# --- Flask Routes ---
@app.before_request
def ensure_data_loaded():
    """Ensures data and model are loaded before any request is processed."""
    if jobs_df is None:
        if not load_and_prepare_data():
            # If loading fails, we might want to render an error page
            # For now, print error and allow the route to handle it
            pass

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    error_message = None

    if request.method == 'POST':
        if 'resume_file' not in request.files:
            error_message = 'No file part'
        else:
            file = request.files['resume_file']
            if file.filename == '':
                error_message = 'No selected file'
            elif file and allowed_file(file.filename):
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(filepath)
                    resume_content = extract_text_from_file(filepath)

                    if not resume_content.strip():
                        error_message = "Could not extract sufficient text from the uploaded resume. Please try another file."
                    else:
                        print(f"Extracted resume content length: {len(resume_content)} chars")
                        # Get hybrid recommendations
                        recommendations_df = get_hybrid_recommendations(resume_content, ir_top_k=20, ml_top_k=5)

                        if not recommendations_df.empty:
                            # Prepare data for HTML table
                            recommendations = recommendations_df[[
                                'Job Title', 'predicted_score', 'Job Description'
                            ]].rename(columns={
                                'predicted_score': 'Match Score',
                                'Job Description': 'Description Snippet'
                            }).to_dict(orient='records')
                            # Truncate description snippet for display
                            for rec in recommendations:
                                rec['Description Snippet'] = rec['Description Snippet'][:200] + "..." if len(rec['Description Snippet']) > 200 else rec['Description Snippet']
                        else:
                            error_message = "No job recommendations found for the uploaded resume."

                except Exception as e:
                    error_message = f"An error occurred during file processing or recommendation: {e}"
                    print(f"Full traceback: {traceback.format_exc()}")
                finally:
                    # Clean up the uploaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)
            else:
                error_message = 'Invalid file type. Allowed: .txt, .pdf, .docx'

    return render_template('index.html', recommendations=recommendations, error_message=error_message)

if __name__ == '__main__':
    # Load data and train model once when the app starts
    if load_and_prepare_data():
        app.run(debug=True, port=5000)
    else:
        print("Application cannot start due to data loading/model training failure.")
