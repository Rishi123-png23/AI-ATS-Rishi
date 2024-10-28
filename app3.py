import streamlit as st
import pdfplumber
import docx
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from dotenv import load_dotenv
import base64
from groq import Groq
import google.generativeai as genai
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import numpy as np
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import requests
import json
from io import BytesIO


nltk.download('punkt', quiet=True) 

#img
img_logo = Image.open("images/img_logo.png")
img_contact_form = Image.open("images/img1.jpeg")
img2 = Image.open("images/img2.png")
img4 = Image.open("images/img4.png")
img_logo1 = Image.open("images/img_logo1.png")




# Page configuration
st.set_page_config(
    page_title="AIonOS-AI-ATS-125073-RK",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color scheme based on AIonOS branding
COLORS = {
    'primary': '#00B8D4',      # AIonOS Primary Blue
    'secondary': '#0288D1',    # Darker Blue
    'accent': '#00BCD4',       # Bright Accent
    'success': '#4CAF50',      # Success Green
    'warning': '#FFC107',      # Warning Yellow
    'error': '#F44336',        # Error Red
    'background': '#F5F7FA',   # Light Background
    'text': '#2C3E50',         # Dark Text
    'light_text': '#607D8B',   # Light Text
    'border': '#E0E6ED',       # Border Color
    'card_bg': '#F0F2F5'       # Light Grey Card Background
}

# Enhanced CSS with AIonOS branding
st.markdown(f"""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {{
        font-family: 'Poppins', sans-serif;
        color: {COLORS['text']};
    }}
    
    /* Main Container */
    .main {{
        background-color: {COLORS['background']};
        padding: 2rem;
    }}
    
    /* Header Styles */
    h1, h2, h3 {{
        color: {COLORS['primary']};
        font-weight: 600;
    }}
    
    /* Custom Header Container */
    .header-container {{
        display: flex;
        align-items: center;
        padding: 1rem 2rem;
        background: white;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 2rem;
    }}
    
    .logo-container {{
        flex: 0 0 auto;
        margin-right: 2rem;
    }}
    
    .nav-container {{
        flex: 1;
    }}
    
    /* Card Styles */
    .stCard {{
        background: {COLORS['card_bg']};
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid {COLORS['border']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }}
    
    /* Metrics Card Style */
    .css-1wivap2 {{
        background-color: {COLORS['card_bg']} !important;
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1rem;
    }}
    
    /* Button Styles */
    .stButton > button {{
        background-color: {COLORS['primary']};
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['secondary']};
        transform: translateY(-2px);
    }}
    
    /* Progress Bar */
    .stProgress > div > div {{
        background-color: {COLORS['success']};
    }}
    
    /* Upload Section */
    .uploadedFile {{
        border: 2px dashed {COLORS['primary']};
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        background: rgba(0,184,212,0.05);
    }}
    
    /* Navigation Menu */
    .nav-link {{
        background: {COLORS['card_bg']};
        margin: 0.2rem 0;
        border-radius: 5px;
        transition: all 0.3s ease;
    }}
    
    .nav-link:hover {{
        background: {COLORS['primary']};
        color: white !important;
    }}
    
    .nav-link-selected {{
        background: {COLORS['primary']} !important;
        color: white !important;
    }}
    
    /* Footer */
    .footer {{
        background: white;
        padding: 1.5rem;
        text-align: center;
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        border-top: 1px solid {COLORS['border']};
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    
    .footer-content {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }}
    
    .footer-left {{
        text-align: left;
    }}
    
    .footer-right {{
        text-align: right;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['background']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['primary']};
        border-radius: 4px;
    }}

    /* Hide Streamlit Footer and Main Menu */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    footer:after {{content: ''; visibility: hidden;}}

</style>
""", unsafe_allow_html=True)

# Load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Professional animations
lottie_analysis = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_hiring = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_xyadoh9h.json")
lottie_chat = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_kq5rGs.json")

# Create header with logo and navigation
# Load and resize logo
try:
    img_logo = Image.open("images/img_logo.png")
    # Resize while maintaining aspect ratio
    logo_width = 150
    aspect_ratio = img_logo.size[1] / img_logo.size[0]
    logo_height = int(logo_width * aspect_ratio)
    img_logo = img_logo.resize((logo_width, logo_height))
except Exception as e:
    st.error(f"Error loading logo: {str(e)}")
    img_logo = None

# Create header container
st.markdown('<div class="header-container">', unsafe_allow_html=True)

# Logo column
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
if img_logo is not None:
    st.image(img_logo)
st.markdown('</div>', unsafe_allow_html=True)

# Navigation column
st.markdown('<div class="nav-container">', unsafe_allow_html=True)
selected = option_menu(
    menu_title=None,
    options=["Home", "ATS Analysis", "RAAG", "AI Assistant"],
    icons=["house-fill", "search", "person-badge-fill", "chat-dots-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": COLORS['primary'], "font-size": "1.2rem"},
        "nav-link": {
            "font-size": "1rem",
            "text-align": "center",
            "margin": "0px",
            "padding": "0.8rem 1rem",
            "--hover-color": COLORS['primary']
        },
        "nav-link-selected": {"background-color": COLORS['primary']}
    }
)
st.markdown('</div>', unsafe_allow_html=True)

# Close header container
st.markdown('</div>', unsafe_allow_html=True)


# Home Page
if selected == "Home":
    st.markdown(f"""
    <div class="stCard" style="text-align: center; padding: 2rem;">
        <h1 style="color: {COLORS['primary']}; font-size: 2.5rem; margin-bottom: 1rem;">
            Welcome to AIonOS Enterprise ATS
        </h1>
        <p style="color: {COLORS['light_text']}; font-size: 1.2rem; margin-bottom: 2rem;">
            Next-Generation Talent Acquisition & Analysis Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Lottie Animation
    st_lottie(lottie_analysis, height=300, key="welcome")
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stCard" style="text-align: center;">
            <h3 style="color: {COLORS['primary']}">🎯 Smart Analysis</h3>
            <p>AI-powered resume analysis with deep insights and matching</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stCard" style="text-align: center;">
            <h3 style="color: {COLORS['primary']}">💡 Intelligent Q&A</h3>
            <p>Get instant answers about any resume with our AI assistant</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stCard" style="text-align: center;">
            <h3 style="color: {COLORS['primary']}">👥 Hiring Assistant</h3>
            <p>Expert guidance for making informed hiring decisions</p>
        </div>
        """, unsafe_allow_html=True)

    with st.container():
                      st.write("-------")
                      st.header("Vision")
                      image_column,text_column = st.columns((1,2))
                      with image_column:
                           st.image(img_contact_form)
       

                      with text_column:
                           st.subheader("Objective")
                           st.write(                                                                                                                                                                                                
                            """
                                  The ATS platform is designed to evaluate the ATS score of resumes against job descriptions,
                                   enhancing the recruitment process for HR professionals. It utilizes AI to provide valuable insights about candidates based on their resumes and features a visualization tool that summarizes candidate information for quick assessment. 
                                   The platform also incorporates large language model (LLM) functionality, enabling HR personnel to upload resumes and ask questions directly related to candidates. 
                                   The AI responds with relevant answers extracted from the resumes, streamlining the review process. Additionally, a dedicated chat assistant is available to address hiring-related queries, such as interview questions and processes, ensuring comprehensive support throughout the recruitment journey. 
                                  This combination of features empowers HR teams to make informed decisions efficiently and effectively.
                            """ 
                                )

# The rest of your existing functions (extract_text_from_pdf, calculate_match_score, etc.)
# Your existing imports and setup code here...


# Load environment variables from .env file
load_dotenv()

# Check if the API key is properly loaded
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set the 'GOOGLE_API_KEY' in your environment.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Clean text by removing stopwords and punctuation
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    cleaned_text = " ".join([word for word in words if word.isalnum() and word not in stop_words])
    return cleaned_text

# Match keywords between resume and job description
def calculate_match_score(resume_text, job_description):
    vectorizer = CountVectorizer().fit([resume_text, job_description])
    resume_vector = vectorizer.transform([resume_text]).toarray()
    job_vector = vectorizer.transform([job_description]).toarray()
    score = (resume_vector * job_vector).sum() / job_vector.sum()
    return round(score * 100, 2)

# Function to generate a downloadable text file
def create_download_link(file_content, filename="ATS_Resume_Report.txt"):
    b64 = base64.b64encode(file_content.encode()).decode()  # encode file content as Base64
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Click here to download your report</a>'
    return href

# Define the experience extraction function
def extract_experience_years(resume_text):
    experience_years = 0
    experience_pattern = r'(\d+)\s*years?'
    
    matches = re.findall(experience_pattern, resume_text)
    if matches:
        experience_years = max(map(int, matches))
    
    return experience_years

def calculate_education_percentage(grade_10, intermediate, graduation, postgraduate=None):
    percentages = {
        "10th": (grade_10 / 10) * 100,
        "Intermediate": (intermediate / 10) * 100,
        "Graduation": (graduation / 10) * 100,
    }

    if postgraduate is not None:
        percentages["Postgraduate"] = (postgraduate / 10) * 100

    total_weight = 3 if postgraduate is None else 4
    overall_percentage = sum(percentages.values()) / total_weight

    return overall_percentage, percentages

def plot_education_distribution(percentages):
    labels = percentages.keys()
    sizes = percentages.values()

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title("Education Percentage Distribution")
    st.pyplot(fig)
# Generate insights using Gemini AI with provided prompt
def get_gemini_response(resume_text, job_description):
    if not GOOGLE_API_KEY:
        return "API key is missing. Unable to call the AI model."
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        The resume text is:

        {resume_text}

        The job description is:

        {job_description}

        Please analyze the resume and job description to provide the following insights and clear information:

        * Missing keywords from the resume
        * Missing skills from the resume and which are present in job description
        * Candidate's strengths based on the resume
        * Candidate's weaknesses based on the resume
        * Highlights of the resume
        * Skills and keywords which are matched in both resume and job description
        * Give me clear information related to relevant skills, education, and work experience aligned with the job description
        * Overall suggestion for the HR regarding the candidate's suitability
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while getting a response: {e}"

# Streamlit App Layout
#st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")


def analyze_resume(resume_text, question):
    if not GOOGLE_API_KEY:
        return "API key is missing. Unable to call the AI model."
    
    try:
        # Generate a response based on the resume text and question
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        The resume text is:

        {resume_text}

        Question: {question}

        Please answer the question based on the resume.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while analyzing the resume: {str(e)}"

# would continue here, but with enhanced styling and professional features...

# Example of how to style the ATS Analysis section:
# Main Streamlit app
if selected == "ATS Analysis":
    st.markdown(f"""
    <div class="stCard">
        <h2 style="color: {COLORS['primary']}">Resume Analysis & Matching</h2>
        <p style="color: {COLORS['light_text']}">Upload a resume and job description for comprehensive analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="stCard">
            <h3 style="color: {COLORS['secondary']}">Document Upload</h3>
            <p>Supported formats: PDF, DOCX</p>
        </div>
        """, unsafe_allow_html=True)
        
        resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
        job_description = st.text_area("Job Description", height=200)
    
    with col2:
        st_lottie(lottie_analysis, height=300, key="analysis")
    
    if st.button("Analyze Resume"):
        if resume_file and job_description:
            try:
                # Adding a loading spinner while processing
                with st.spinner("Analyzing resume..."):
                    # Extracting and cleaning resume and job description
                    resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
                    cleaned_resume = clean_text(resume_text)
                    cleaned_job_desc = clean_text(job_description)
                    
                    # Calculate match score and generate AI insights
                    match_score = calculate_match_score(cleaned_resume, cleaned_job_desc)
                    insights = get_gemini_response(cleaned_resume, cleaned_job_desc)
                    experience_years = extract_experience_years(cleaned_resume)

                # Display the ATS Score and AI Insights in a single section
                st.subheader("📊 ATS Score and AI Insights")
                st.progress(match_score / 100)
                st.metric(label="Match Score", value=f"{match_score}%")
                st.subheader("🤖 Gemini AI Insights")
                st.info(insights)

                # Combined Visualization
                st.markdown("### Combined Visualizations")

                # Normalize experience years to a scale of 0 to 100 with a max value
                max_experience_years = 20  # Set max experience years to 20 for better normalization
                normalized_experience = (experience_years / max_experience_years) * 100

                fig, ax = plt.subplots(figsize=(10, 5))

                # Match Score Bar
                ax.barh(["Resume Match"], [match_score], color='skyblue', label='Match Score')
                ax.set_xlim(0, 100)
                ax.set_xlabel('Score (%)')
                ax.set_title('Resume Match Score and Experience Meter')

                # Adding Experience Years as a separate bar with a minimum threshold for visualization
                experience_bar_value = max(normalized_experience, 5)  # Ensure bar is visible with a min value of 5
                ax.barh(["Years of Experience"], [experience_bar_value], color='orange', label='Years of Experience (normalized)')
                
                # Adding legends
                ax.legend(loc='upper right')

                st.pyplot(fig)

                # Prepare content for the downloadable report
                report_content = f"""
                ATS Resume Checker Report
                -------------------------
                Match Score: {match_score}%
                
                Resume Text:
                {resume_text}
                
                Job Description:
                {job_description}
                
                Gemini AI Insights:
                {insights}
                """

                # Provide a downloadable file button in the UI
                st.download_button(
                    label="📥 Download ATS Report",
                    data=report_content,
                    file_name="ATS_Resume_Report.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a resume and provide a job description.")
# Continue with the rest of your sections...
# Remember to maintain consistent styling throughout the application

elif selected == "RAAG":
    st.markdown("""
    <div class="modern-card gradient-bg">
        <h1>🤖 Retrieval-Augmented Answer Generation</h1>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
                      st.write("-------")
                      image_column,text_column = st.columns((1,2))
                      with image_column:
                           st.image(img2)
       
                      with text_column:
                           st.write(                                                                                                                                                                                                
                            """
                                 The RAAG (Retrieval-Augmented Answer Generation) model, powered by Gemini LLM, provides an interactive solution for knowledge extraction. 
                                 With a light peacock blue interface, it allows users to upload PDFs and text documents directly from their devices. 
                                 Users can then ask specific questions related to these documents, and the LLM draws from both its trained knowledge and the uploaded content to generate accurate, relevant answers. 
                                 This setup combines retrieval-based data sourcing with answer generation, making it ideal for applications in research, customer support, and knowledge management. 
                                 
                            """ 
                                )
    st.subheader("Ask questions about any resume and get AI-powered answers")
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "doc", "docx"])
    
    # Placeholder for resume text
    resume_text = ""

    if resume_file is not None:
        # Extract the resume text
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            resume_text = extract_text_from_docx(resume_file)

        st.success("Resume uploaded successfully!")

        # Question input box
        question = st.text_input("Ask a question about the resume:")

        if st.button("Submit Question"):
            if question:
                # Show spinner while processing the answer
                with st.spinner("Analyzing your question..."):
                    answer = analyze_resume(resume_text, question)  # Call the function to analyze the resume
                    st.success("Answer: {}".format(answer))  # Display the answer
            else:
                st.error("Please enter a question.")



   # Your existing RAAG code with enhanced UI...

# Hiring Assistant Section
elif selected == "AI Assistant":
    st.markdown("""
    <div class="modern-card gradient-bg">
        <h1>👥 AI Hiring Assistant 🧠 </h1>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
                      st.write("-------")
                      st.header("Assistant")
                      image_column,text_column = st.columns((1,2))
                      with image_column:
                           st.image(img4)
       
                      with text_column:
                           st.write(                                                                                                                                                                                                
                            """
                                 This AI Assistant is designed as a knowledgeable companion for hiring managers, providing expert support throughout the recruitment process. 
                                 With an intuitive, light peacock blue interface, it offers insights on candidate evaluations, tailored interview questions, and strategic hiring decisions. 
                                 Users can ask specific questions, receive expert advice, and gain actionable recommendations to evaluate resumes, select relevant interview questions, and match candidates to job descriptions effectively. 
                                 Powered by the robust LLaMA3-70b-8192 model from Groq Cloud, the assistant ensures high-quality guidance tailored to each hiring need, making it a valuable tool for streamlined, informed recruitment.
                                 
                            """ 
                                )
    
    # Your existing Hiring Assistant code...
    # Initialize Groq API
    GROQ_API_KEY = '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'
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # System prompt for the assistant
    system_prompt = {
        "role": "system",
        "content": (
            "You are a knowledgeable hiring manager's assistant designed to help users with candidate evaluations, "
            "interview questions, and hiring process inquiries. Assist with providing insights about resumes, "
            "suggest interview questions, and evaluate candidates based on job descriptions and their qualifications. "
            "Provide clear, actionable advice based on the user's needs."
        )
    }

    # Streamlit app setup
    prompt = st.text_input('Ask your research question:')

    # Add a submit button
    if st.button('Submit'):
        if prompt:
            chat_history = [system_prompt, {"role": "user", "content": prompt}]
            
            try:
                # Groq model call
                response = client.chat.completions.create(model="llama3-70b-8192", messages=chat_history)
                
                # Display the assistant's response
                st.write(response.choices[0].message.content)

            except ConnectionError:
                st.error("Connection error: Please check your internet connection and try again.")
            except TimeoutError:
                st.error("Request timed out. Please try again later.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
        else:
            st.warning("Please enter a question before submitting.")


#footer
# Function to convert PIL image to base64
def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load and resize logo for footer (make it smaller)
footer_logo = img_logo.copy()
footer_logo = footer_logo.resize((100, 35))  # Adjust size as needed
logo_base64 = img_to_base64(footer_logo)

# Updated footer CSS
st.markdown("""
<style>
    .footer {
        background-color: #f5f5f5;
        padding: 0.5rem 2rem;
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        font-size: 0.875rem;
        border-top: 1px solid #e0e0e0;
        height: 60px;  /* Reduced height */
    }
    
    .footer-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .footer-left {
        display: flex;
        align-items: center;
        gap: 2rem;
    }
    
    .footer-logo img {
        height: 30px;
        width: auto;
    }
    
    .contact-button {
        background-color: #00B8D4;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 4px;
        text-decoration: none;
        font-size: 0.875rem;
        transition: background-color 0.3s;
    }
    
    .contact-button:hover {
        background-color: #0288D1;
    }
    
    .footer-right {
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Footer HTML content
footer_html = f"""
<div class="footer">
    <div class="footer-content">
        <div class="footer-left">
            <div class="footer-logo">
                <img src="data:image/png;base64,{logo_base64}" alt="AIonOS Logo">
            </div>
            <a href="https://aionos.io" target="_blank" class="contact-button">Contact Us</a>
        </div>
        <div class="footer-right">
            © 2024 AIonOS. All Rights Reserved.
        </div>
    </div>
</div>
"""

# Add the footer to your Streamlit app
st.markdown(footer_html, unsafe_allow_html=True)
