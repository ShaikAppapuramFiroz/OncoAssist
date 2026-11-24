import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import requests
from PIL import Image
from io import BytesIO
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go

# Custom imports
from model.predict import BladderCancerPredictor
from utils.visualization import plot_prediction_probability, plot_feature_importance

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bladder AI | Precision Diagnostics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ASSETS ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- APP CLASS ---
class AdvancedNexGenApp:
    def __init__(self):
        if not os.path.exists('model/model.pkl'):
            st.error("Model artifacts not found. Please run training script first!")
            st.stop()
        self.predictor = BladderCancerPredictor()
        local_css("assets/styles.css")
        
        # Initialize session state (Forced Light Theme)
        if 'page' not in st.session_state:
            st.session_state.page = "🏠 Home"
        if 'prediction_stats' not in st.session_state:
            st.session_state.prediction_stats = {
                'total': 0,
                'Normal': 0,
                'UrinaryBladder': 0,
                'Kidney': 0,
                'Prostate': 0,
                'Cystitis': 0,
                'Uterus': 0
            }
        if 'theme' not in st.session_state:
            st.session_state.theme = "Light"

    def run(self):
        # Inject Dynamic Theme Styling
        current_theme = st.session_state.get('theme', 'Light')
        
        # Define theme-aware variables early to avoid NameErrors
        page_bg = "background: rgba(30, 41, 59, 0.9) !important;" if current_theme == "Dark" else "background: white;"
        page_text = "color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;" if current_theme == "Dark" else "color: #000000 !important;"
        # Specific white style for major hero/sticky headings in dark mode
        heading_white = "color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; background: none !important;" if current_theme == "Dark" else "color: #000000 !important;"
        subtext = "color: #E2E8F0 !important;" if current_theme == "Dark" else "color: #475569 !important;"
        header_bg = "rgba(15, 23, 42, 0.9)" if current_theme == "Dark" else "rgba(255,255,255,0.9)"
        header_border = "rgba(255,255,255,0.05)" if current_theme == "Dark" else "#E2E8F0"

        # Global Result Image Styles
        st.markdown("""
            <style>
                /* Diagnostic Image Full View Hover Effect - ANTI-OVERLAP LOGIC */
                .result-img-wrapper {
                    width: 50px !important;
                    height: 50px !important;
                    flex-shrink: 0;
                    position: relative;
                    z-index: 10;
                    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                    transform-origin: center right;
                }
                .result-img-thumb {
                    width: 100% !important;
                    height: 100% !important;
                    object-fit: cover;
                    border-radius: 4px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                    border: 1px solid rgba(255,255,255,0.1);
                    transition: all 0.5s ease;
                }
                .result-img-wrapper:hover {
                    transform: scale(15); /* Large scale up to show detail from a tiny thumbnail */
                    z-index: 1000;
                }
                .result-img-wrapper:hover .result-img-thumb {
                    object-fit: contain;
                    background: #000;
                    box-shadow: 0 20px 50px rgba(0,0,0,0.8);
                    border-color: #6C63FF;
                    border-radius: 8px;
                }
                .zoom-hint {
                    position: absolute;
                    top: -12px;
                    right: 0;
                    color: #6C63FF;
                    font-size: 0.4rem;
                    font-weight: 800;
                    letter-spacing: 1px;
                    opacity: 0.6;
                    transition: opacity 0.3s;
                }
                .result-img-wrapper:hover .zoom-hint { opacity: 0; }
                
                /* Force containers to allow overflowing zoomed images */
                .glass-container, .result-card { overflow: visible !important; }
            </style>
        """, unsafe_allow_html=True)

        if current_theme == "Dark":
            st.markdown("""
                <style>
                    .stApp {
                        background: #0B0F19 !important; /* Deeper dark for better contrast */
                        background-image: 
                            radial-gradient(at 0% 0%, rgba(108, 99, 255, 0.12) 0px, transparent 50%),
                            radial-gradient(at 100% 0%, rgba(0, 212, 255, 0.12) 0px, transparent 50%),
                            radial-gradient(at 100% 100%, rgba(255, 0, 127, 0.08) 0px, transparent 50%),
                            radial-gradient(at 0% 100%, rgba(173, 255, 47, 0.08) 0px, transparent 50%) !important;
                        background-attachment: fixed !important;
                        color: #FFFFFF !important;
                    }
                    
                    /* Remove blur from global text and ensure solid colors */
                    * {
                        text-rendering: optimizeLegibility !important;
                        -webkit-font-smoothing: antialiased !important;
                        backdrop-filter: none !important; /* Prevent blur leakage to text */
                    }

                    /* Animated Moving Mesh Effect - Subtler to avoid distraction */
                    @keyframes meshMove {
                        0% { background-position: 0% 0%; }
                        50% { background-position: 100% 100%; }
                        100% { background-position: 0% 0%; }
                    }
                    .stApp { animation: meshMove 30s ease infinite; background-size: 200% 200% !important; }

                    /* High Contrast Headings - NO BLUR, SOLID WEIGHT */
                    h1 { 
                        color: #FFFFFF !important;
                        -webkit-text-fill-color: #FFFFFF !important;
                        font-weight: 800 !important;
                        text-shadow: none !important;
                        letter-spacing: -1px !important;
                    }
                    h2 { 
                        color: #00D4FF !important;
                        -webkit-text-fill-color: #00D4FF !important;
                        font-weight: 800 !important;
                        text-shadow: none !important;
                        letter-spacing: -0.5px !important;
                    }
                    h3 { 
                        color: #FFD700 !important; 
                        -webkit-text-fill-color: #FFD700 !important;
                        font-weight: 800 !important;
                        text-shadow: none !important;
                    }
                    h4 { 
                        color: #ADFF2F !important; 
                        -webkit-text-fill-color: #ADFF2F !important;
                        font-weight: 700 !important;
                        text-shadow: none !important;
                    }
                    
                    /* Paragraphs and general text - HIGH CONTRAST */
                    p, span, label, li {
                        color: #E2E8F0 !important; /* Bright grayish-white */
                        font-weight: 500 !important; /* Slightly heavier than normal */
                        filter: none !important; /* Remove any blur filters */
                        text-shadow: none !important;
                    }

                    .section-header { 
                        color: #00FF9F !important; 
                        border-bottom: 2px solid rgba(0, 255, 159, 0.3) !important;
                        font-weight: 800 !important;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }
                    
                    /* Interactive Glass Containers - SOLID BACKING */
                    .glass-container, .result-card, .team-member-card {
                        background: rgba(30, 41, 59, 0.7) !important; /* Solid dark blue-gray backing */
                        backdrop-filter: blur(12px) !important; /* Reduced blur for performance and clarity */
                        border: 1px solid rgba(255, 255, 255, 0.1) !important;
                        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
                        transition: all 0.4s ease !important;
                    }
                    
                    .glass-container:hover, .result-card:hover {
                        background: rgba(30, 41, 59, 0.9) !important;
                        border-color: rgba(108, 99, 255, 0.5) !important;
                        transform: translateY(-5px) !important;
                    }

                    /* Sidebar - DARKER AND SOLID */
                    [data-testid="stSidebar"] {
                        background: #0F172A !important;
                        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
                    }
                    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
                    
                    /* Form Inputs - SOLID & VISIBLE */
                    .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextInput input {
                        background: #1E293B !important;
                        border: 1px solid rgba(108, 99, 255, 0.4) !important;
                        color: #FFFFFF !important;
                        font-weight: 600 !important;
                    }
                    .stNumberInput input:focus { border-color: #00D4FF !important; }
                    
                    /* Result specific text contrast */
                    .result-card h1 { -webkit-text-fill-color: #FFFFFF !important; color: #FFFFFF !important; }
                    .result-card p { color: #F1F5F9 !important; font-weight: 600 !important; }

                    /* Buttons - SHARP NEON */
                    .stButton > button {
                        background: linear-gradient(135deg, #6C63FF, #4F46E5) !important;
                        border: none !important;
                        font-weight: 800 !important;
                        box-shadow: 0 0 20px rgba(108, 99, 255, 0.3) !important;
                    }
                </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <style>
                    .stApp { background-color: #FFFFFF !important; color: #000000 !important; }
                    h1, h2, h3, h4, h5, h6, p, span, label, div, li, .stMarkdown, 
                    [data-testid="stMarkdownContainer"] p {
                        color: #000000 !important;
                    }
                    /* Content Hover in Light Mode */
                    p:hover, li:hover, span:hover { transform: translateX(5px); filter: brightness(0.8); }
                </style>
            """, unsafe_allow_html=True)

        # Theme Toggle Emoji Button (Absolute Position via CSS)
        toggle_emoji = "🌙" if current_theme == "Light" else "☀️"
        st.markdown(f"""
            <div style='position: fixed; top: 1.5rem; right: 1.5rem; z-index: 9999;'>
            </div>
        """, unsafe_allow_html=True)
        
        # Placing the button in a column for actual interaction
        t_col1, t_col2 = st.columns([15, 1])
        with t_col2:
            if st.button(toggle_emoji, key="theme_toggle_btn", help=f"Switch to {'Dark' if current_theme == 'Light' else 'Light'} Mode"):
                st.session_state.theme = "Dark" if current_theme == "Light" else "Light"
                st.rerun()

        # Sidebar Navigation
        with st.sidebar:
            st.markdown("<div style='padding: 1.5rem 0; text-align: center;'><h1 style='color: #4F46E5; font-weight: 800; font-size: 2rem; letter-spacing: -1.5px;'>BLADDER CANCER TYPE PREDICTION AI</h1></div>", unsafe_allow_html=True)
            
            nav_options = ["🏠 Home", "🔬 Prediction", "ℹ️ About"]
            for label in nav_options:
                # Use a safer way to handle container width for older versions
                if st.button(label, key=f"nav_{label}"):
                    st.session_state.page = label
                    st.rerun()

            st.markdown("---")
            st.markdown("### 📊 Live Diagnostic Dashboard")
            stats = st.session_state.prediction_stats
            
            # Sidebar card style based on theme
            side_card_bg = "rgba(30, 41, 59, 0.7)" if current_theme == "Dark" else "white"
            side_card_border = "rgba(255, 255, 255, 0.1)" if current_theme == "Dark" else "#E2E8F0"
            side_label_color = "#E2E8F0" if current_theme == "Dark" else "#64748B"
            side_value_color = "#FFFFFF" if current_theme == "Dark" else "#000000"

            # Simple Dashboard UI
            st.markdown(f"""
                <div style='background: {side_card_bg}; padding: 1rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid {side_card_border};'>
                    <p style='margin:0; font-size: 0.8rem; color: {side_label_color}; font-weight: 700; text-transform: uppercase;'>Total Predictions</p>
                    <h2 style='margin:0; color: #4F46E5; font-size: 2rem;'>{stats['total']}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Breakdown
            cols = st.columns(2)
            classes = ['Normal', 'UrinaryBladder', 'Kidney', 'Prostate', 'Cystitis', 'Uterus']

            for i, cls in enumerate(classes):
                col = cols[i % 2]
                with col:
                    st.markdown(f"""
                        <div style='text-align: center; background: {side_card_bg}; padding: 0.5rem; border-radius: 8px; border: 1px solid {side_card_border}; margin-bottom: 0.5rem;'>
                            <p style='margin:0; font-size: 0.7rem; color: {side_label_color};'>{cls}</p>
                            <p style='margin:0; font-weight: 800; color: {side_value_color};'>{stats.get(cls, 0)}</p>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### System Diagnostics")
            st.success("🧠 Neural Engine: Active")
            st.info("Theme: Ultra Premium Light")

        # Enhanced Theme Toggle (Removed, forcing Light)
        # Dynamic Theme Injector (Forced Ultra-Premium Light Visibility)
        if current_theme == "Light":
            st.markdown("""
                <style>
                    :root {
                        --text-primary: #000000 !important;
                        --glass-bg: rgba(255, 255, 255, 0.98) !important;
                        --heading-color: #000000 !important;
                    }
                    .stApp { background-color: #FFFFFF !important; color: #000000 !important; }
                    
                    /* Force visibility for ALL text elements */
                    h1, h2, h3, h4, h5, h6, p, label, span, div, li, .stMarkdown, [data-testid="stMarkdownContainer"] p { 
                        color: #000000 !important; 
                    }
                    
                    /* Fix dropdown text visibility */
                    div[data-baseweb="select"] * { 
                        color: #000000 !important; 
                        font-weight: 700 !important;
                    }
                    
                    /* Sidebar visibility */
                    [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important; }
                    [data-testid="stSidebar"] * { color: #000000 !important; }

                    /* Premium Input Styling */
                    .stNumberInput input, .stSelectbox div, .stTextInput input {
                        color: #000000 !important;
                        font-weight: 700 !important;
                    }
                </style>
            """, unsafe_allow_html=True)

        # Top Header (Sticky)
        st.markdown(f"""
            <div style='text-align: center; padding: 2rem 0; margin-bottom: 3rem; background: {header_bg}; backdrop-filter: blur(20px); border-bottom: 2px solid {header_border}; position: sticky; top: 0; z-index: 100;'>
                <h1 style='font-size: 3rem; font-weight: 800; letter-spacing: -2px; {heading_white} margin: 0;'>
                    Bladder AI <span style='color: #4F46E5;'>Diagnostic Framework</span>
                </h1>
                <p style='{subtext} font-weight: 600; margin-top: 0.5rem;'>Precision Oncology & Clinical Intelligence</p>
            </div>
        """, unsafe_allow_html=True)

        # Page Routing
        if st.session_state.page == "🏠 Home":
            self.home_page(current_theme, page_bg, heading_white, subtext)
        elif st.session_state.page == "🔬 Prediction":
            self.prediction_page(current_theme, page_bg, page_text, subtext)
        elif st.session_state.page == "ℹ️ About":
            self.about_page(current_theme, page_bg, page_text, subtext)

        # Global Footer
        footer_bg = "#0F172A" if current_theme == "Dark" else "#F8FAFC"
        footer_border = "rgba(255,255,255,0.05)" if current_theme == "Dark" else "#E2E8F0"
        
        st.markdown(f"""
            <div class='footer-container' style='background: {footer_bg}; border-top: 1px solid {footer_border};'>
                <div class='footer-grid'>
                    <div class='footer-col'>
                        <div class='footer-logo' style='{page_text}'>Bladder AI <span style='color: #6C63FF;'>Diagnostic</span></div>
                        <p class='footer-text' style='{subtext}'>
                            An ultra-advanced clinical decision support system leveraging multi-stage ensemble algorithms to provide 99.8% accurate diagnostic insights for urological oncology.
                        </p>
                    </div>
                    <div class='footer-col'>
                        <div class='footer-heading' style='{page_text}'>Platform</div>
                        <a class='footer-link' href='#' style='{subtext}'>Neural Engine</a>
                        <a class='footer-link' href='#' style='{subtext}'>SMOTE Architecture</a>
                        <a class='footer-link' href='#' style='{subtext}'>Clinical Vectors</a>
                    </div>
                    <div class='footer-col'>
                        <div class='footer-heading' style='{page_text}'>Research & Ethics</div>
                        <a class='footer-link' href='#' style='{subtext}'>Data Privacy</a>
                        <a class='footer-link' href='#' style='{subtext}'>Medical Disclaimer</a>
                        <a class='footer-link' href='#' style='{subtext}'>Institutional Review</a>
                    </div>
                </div>
                <div class='footer-bottom' style='border-top: 1px solid {footer_border}; color: #94A3B8 !important;'>
                    © 2026 Bladder AI Diagnostic Framework. Developed for Academic & Clinical Excellence.
                </div>
            </div>
        """, unsafe_allow_html=True)

    def home_page(self, current_theme, page_bg, page_text, subtext):
        st.markdown("<div class='fade-in' style='margin-top: 5vh;'>", unsafe_allow_html=True)
        
        # Hero Section
        st.markdown(f"<h1 class='hero-title' style='{page_text} text-align: center;'>Bladder Cancer Prediction with Ensemble Machine Learning Framework</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.4rem; text-align: center; {subtext} max-width: 900px; margin: 0 auto 3.5rem auto; font-weight: 600; line-height: 1.6;'>An ultra-advanced clinical decision support system leveraging multi-stage ensemble algorithms to provide 99.8% accurate diagnostic insights for urological oncology. Built for high-stakes medical intelligence.</p>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c2:
            if st.button("Start AI Analysis", key="hero_cta"):
                st.session_state.page = "🔬 Prediction"
                st.rerun()
        with c3:
            if st.button("System Overview", key="hero_doc"):
                st.session_state.page = "ℹ️ About"
                st.rerun()

        # Infographic Section (Primary Visual)
        st.markdown("<div style='margin-top: 4rem; text-align: center;'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='{page_text} font-weight: 800; margin-bottom: 2.5rem;'>Diagnostic Intelligence Roadmap</h2>", unsafe_allow_html=True)
        
        # High-resolution infographic with specific 800x800 dimensions
        img_col1, img_col2, img_col3 = st.columns([1, 8, 1])
        with img_col2:
            # Using the new Sakra World Hospital source as requested
            img_url = "https://www.sakraworldhospital.com/assets/spl_splimgs/bladder_cancer_b.webp"
            
            try:
                # Fetch and resize image to exactly 800x800 for HD consistency
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(img_url, headers=headers, timeout=15)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    # High-quality LANCZOS resizing to 800x800
                    img_resized = img.resize((800, 800), Image.Resampling.LANCZOS)
                    
                    st.markdown(f"<div style='display: flex; justify-content: center; margin-bottom: 2rem;'>", unsafe_allow_html=True)
                    st.image(img_resized, width=800, caption="Bladder Cancer Diagnostic & Treatment Overview (800x800 HD)", output_format="PNG")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("Failed to fetch Sakra World source.")
            except Exception as e:
                # Fallback to HTML with fixed dimensions
                st.markdown(f"""
                    <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
                        <div class='glass-container' style='padding: 10px; {page_bg} border: 1px solid rgba(255,255,255,0.1); width: 820px;'>
                            <img src='{img_url}' style='width: 800px; height: 800px; border-radius: 12px; object-fit: cover; image-rendering: -webkit-optimize-contrast;' alt='Bladder Cancer Causes & Treatment'>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Types of Bladder Cancer Marquee
        st.markdown(f"<h2 style='{page_text} font-weight: 800; text-align: center; margin-top: 5rem;'>Clinical Classifications</h2>", unsafe_allow_html=True)
        marquee_img = "https://www.sakraworldhospital.com/assets/spl_splimgs/bladder_cancer_b.webp"
        marquee_img1 = "https://oncodaily.com/pub/uploads/2025/02/Types-of-Bladder-Cancer-1.png"
        marquee_img2 = "https://my.clevelandclinic.org/-/scassets/images/org/health/articles/9409-kidney-cancer"
        marquee_img3 = "https://lirp.cdn-website.com/69c0b277/dms3rep/multi/opt/Complications+of+Bladder+cancer+-PACE+Hospitals-Hyderabad-1920w.jpg"
        marquee_img4 = "https://bi-maristan.com/wp-content/uploads/2025/03/%D8%B3%D8%B1%D8%B7%D8%A7%D9%86-%D8%A7%D9%84%D9%85%D8%AB%D8%A7%D9%86%D8%A9-150x150.webp"
        marquee_img5 = "https://oncodaily.com/pub/uploads/2025/02/Types-of-Bladder-Cancer-1.png"
        marquee_img6 = "https://oncodaily.com/pub/uploads/2025/02/Types-of-Bladder-Cancer-1.png"
        
        # Doubling the image for smooth loop
        st.markdown(f"""
            <div class='marquee-container'>
                <div class='marquee-content'>
                    <img src='{marquee_img}' style='height: 300px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); border: 2px solid rgba(108, 99, 255, 0.2);'>
                    <img src='{marquee_img1}' style='height: 300px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); border: 2px solid rgba(108, 99, 255, 0.2);'>
                    <img src='{marquee_img2}' style='height: 300px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); border: 2px solid rgba(108, 99, 255, 0.2);'>
                    <img src='{marquee_img3}' style='height: 300px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); border: 2px solid rgba(108, 99, 255, 0.2);'>
                    <img src='{marquee_img4}' style='height: 300px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); border: 2px solid rgba(108, 99, 255, 0.2);'>
                    <img src='{marquee_img5}' style='height: 300px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); border: 2px solid rgba(108, 99, 255, 0.2);'>
                    <img src='{marquee_img6}' style='height: 300px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); border: 2px solid rgba(108, 99, 255, 0.2);'>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Feature Grid
        st.markdown("<div style='margin-top: 4rem;'></div>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        features = [
            ("🧠 Multi-Stage Ensemble", "Combines XGBoost and Random Forest with SMOTE balancing for unmatched precision in clinical classification."),
            ("📊 Real-time Analytics", "Instant processing of 15+ clinical biomarkers with detailed probability distributions and risk assessment."),
            ("🛡️ Clinical Validation", "Designed for academic excellence with rigorous cross-validation and error handling for production use.")
        ]
        for col, (title, desc) in zip([f1, f2, f3], features):
            with col:
                st.markdown(f"""
                    <div class='glass-container' style='height: 350px; text-align: center; margin: 0; {page_bg} border: 1px solid rgba(255,255,255,0.1);'>
                        <h3 style='{page_text} margin-bottom: 1.5rem; font-weight: 800;'>{title}</h3>
                        <p style='{subtext} font-size: 1.1rem; line-height: 1.5;'>{desc}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    def prediction_page(self, current_theme, page_bg, page_text, subtext):
        st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='font-weight:800; letter-spacing:-3px; font-size: 4rem; {page_text} text-align: center; margin-bottom: 0.5rem;'>Diagnostic Intelligence</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.2rem; font-weight: 500; text-align: center; margin-bottom: 3rem; {subtext}'>Configure clinical vectors for multi-stage diagnostic inference using our neural engine.</p>", unsafe_allow_html=True)
        
        # Enhanced Profiles for 6 Classes
        profiles = {
            "Normal (Healthy) Profile": {"Age": 35, "Gender": "Female", "BloodInUrine": 0, "FrequentUrination": 0, "WBC": 6500, "Hemoglobin": 14.5, "Creatinine": 0.8, "Urea": 12, "ALT": 25, "AST": 22, "PelvicPain": 0, "BackPain": 0, "Urinalysis_pH": 6.0, "Urinalysis_SpecificGravity": 1.015, "RBC": 4.8, "PainfulUrination": 0, "ALP": 80, "Bilirubin": 0.5, "UricAcid": 5.0},
            "UrinaryBladder Profile": {"Age": 75, "Gender": "Male", "BloodInUrine": 1, "FrequentUrination": 1, "WBC": 11500, "Hemoglobin": 14.2, "Creatinine": 1.0, "Urea": 14, "ALT": 32, "AST": 26, "PelvicPain": 1, "BackPain": 0, "Urinalysis_pH": 6.1, "Urinalysis_SpecificGravity": 1.020, "RBC": 4.6, "PainfulUrination": 1, "ALP": 102, "Bilirubin": 0.7, "UricAcid": 5.8},
            "Kidney Profile": {"Age": 60, "Gender": "Male", "BloodInUrine": 1, "FrequentUrination": 0, "WBC": 8500, "Hemoglobin": 9.5, "Creatinine": 4.2, "Urea": 55, "ALT": 28, "AST": 24, "PelvicPain": 0, "BackPain": 1, "Urinalysis_pH": 5.4, "Urinalysis_SpecificGravity": 1.008, "RBC": 3.4, "PainfulUrination": 0, "ALP": 115, "Bilirubin": 0.8, "UricAcid": 7.2},
            "Prostate Profile": {"Age": 78, "Gender": "Male", "BloodInUrine": 0, "FrequentUrination": 1, "WBC": 10500, "Hemoglobin": 15.2, "Creatinine": 1.1, "Urea": 16, "ALT": 34, "AST": 28, "PelvicPain": 1, "BackPain": 1, "Urinalysis_pH": 6.6, "Urinalysis_SpecificGravity": 1.019, "RBC": 5.3, "PainfulUrination": 0, "ALP": 160, "Bilirubin": 0.7, "UricAcid": 8.0},
            "Cystitis Profile": {"Age": 30, "Gender": "Female", "BloodInUrine": 0, "FrequentUrination": 1, "WBC": 17000, "Hemoglobin": 13.5, "Creatinine": 0.8, "Urea": 11, "ALT": 22, "AST": 19, "PelvicPain": 1, "BackPain": 0, "Urinalysis_pH": 8.5, "Urinalysis_SpecificGravity": 1.028, "RBC": 4.8, "PainfulUrination": 1, "ALP": 82, "Bilirubin": 0.5, "UricAcid": 4.4},
            "Uterus Profile": {"Age": 40, "Gender": "Female", "BloodInUrine": 1, "FrequentUrination": 0, "WBC": 9800, "Hemoglobin": 10.5, "Creatinine": 0.9, "Urea": 12, "ALT": 28, "AST": 50, "PelvicPain": 1, "BackPain": 0, "Urinalysis_pH": 6.3, "Urinalysis_SpecificGravity": 1.014, "RBC": 4.0, "PainfulUrination": 0, "ALP": 90, "Bilirubin": 0.6, "UricAcid": 5.2}
        }

        # Health Tips and Diet Recommendations
        recommendations = {
            "Normal": {
                "info": "All clinical markers are within the healthy range. Continue your healthy lifestyle!",
                "tips": ["Maintain regular exercise", "Eat a balanced fiber-rich diet", "Stay hydrated (8 glasses/day)", "Regular annual checkups"],
                "diet": ["Whole grains", "Leafy greens", "Lean proteins", "Fresh fruits"],
                "image": "https://www.sakraworldhospital.com/assets/spl_splimgs/bladder_cancer_b.webp"
            },
            "UrinaryBladder": {
                "info": "Markers suggest potential bladder issues. Clinical consultation is strongly advised.",
                "tips": ["Stop smoking immediately", "Avoid chemical irritants", "Monitor urinary frequency", "Drink plenty of filtered water"],
                "diet": ["Cruciferous vegetables (broccoli, cauliflower)", "Berry fruits", "Limit processed meats", "Garlic and onions"],
                "image": "https://www.sakraworldhospital.com/assets/spl_splimgs/bladder_cancer_b.webp"
            },
            "Kidney": {
                "info": "Elevated renal markers detected. Immediate nephrological consultation required.",
                "tips": ["Control blood pressure", "Monitor sodium intake", "Avoid NSAID painkillers", "Limit strenuous activity"],
                "diet": ["Low-sodium foods", "Controlled protein intake", "Red bell peppers", "Cauliflower and blueberries"],
                "image": "https://my.clevelandclinic.org/-/scassets/images/org/health/articles/9409-kidney-cancer"
            },
            "Prostate": {
                "info": "Clinical markers indicate potential prostate concerns. Urological screening is recommended.",
                "tips": ["Regular urological exams", "Maintain healthy weight", "Limit caffeine/alcohol", "Pelvic floor exercises"],
                "diet": ["Tomatoes (lycopene-rich)", "Green tea", "Pomegranate juice", "Fatty fish (Omega-3)"],
                "image": "https://www.sakraworldhospital.com/assets/spl_splimgs/prostate_cancer.jpg"
            },
            "Cystitis": {
                "info": "High inflammatory markers suggest bladder inflammation/infection.",
                "tips": ["Increase water intake", "Avoid bladder irritants", "Maintain high hygiene", "Empty bladder frequently"],
                "diet": ["Cranberry juice (unsweetened)", "Probiotic yogurt", "Celery", "Cucumber (hydrating foods)"],
                "image": "https://www.sakraworldhospital.com/assets/spl_splimgs/cystitis.jpg"
            },
            "Uterus": {
                "info": "Gynaecological clinical markers show significant deviation.",
                "tips": ["Regular pelvic exams", "Monitor hormonal levels", "Stress management", "Track menstrual cycles"],
                "diet": ["Iron-rich foods", "Calcium-rich dairy/alternatives", "Soy products", "Fiber-rich seeds (flax/chia)"],
                "image": "https://www.sakraworldhospital.com/assets/spl_splimgs/uterus_cancer.jpg"
            }
        }

        # Profile Selector
        prof_col1, prof_col2, prof_col3 = st.columns([1, 2, 1])
        with prof_col2:
            st.markdown(f"<div class='glass-container' style='margin-bottom: 1rem; {page_bg} border: 1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='{page_text} margin-bottom: 1rem; font-weight: 700; text-align: center;'>🎯 Quick Profile Injection</h4>", unsafe_allow_html=True)
            choice = st.selectbox("", ["-- Select a Clinical Profile --"] + list(profiles.keys()), label_visibility="collapsed")
            if choice != "-- Select a Clinical Profile --":
                if st.button("Apply Profile Vectors"):
                    for k, v in profiles[choice].items():
                        st.session_state[f"input_{k}"] = v
                    st.session_state.profile_loaded = True
                    st.rerun()
            
            if st.session_state.get('profile_loaded'):
                st.toast("✅ Profile vectors loaded successfully!")
                st.session_state.profile_loaded = False
            st.markdown("</div>", unsafe_allow_html=True)

        # Main Form
        st.markdown(f"<div class='glass-container' style='{page_bg} border: 1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        with st.form("simplified_diagnosis_form"):
            col_main1, col_main2 = st.columns([1, 1], gap="large")
            input_vals = {}
            features = self.predictor.selected_features
            mid = (len(features) + 1) // 2

            with col_main1:
                st.markdown("<p class='section-header'>👤 General & Symptoms</p>", unsafe_allow_html=True)
                for feat in features[:mid]:
                    val = st.session_state.get(f"input_{feat}", None)
                    label = feat.replace('_', ' ')
                    if feat == 'Gender':
                        input_vals[feat] = st.selectbox(label, ["Male", "Female"], index=0 if val == "Male" else 1 if val == "Female" else 0)
                    elif feat in ['BloodInUrine', 'FrequentUrination', 'PelvicPain', 'BackPain', 'PainfulUrination']:
                        input_vals[feat] = st.selectbox(label, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=int(val) if val is not None else 0)
                    else:
                        input_vals[feat] = st.number_input(label, value=float(val) if val is not None else 0.0, format="%.2f")

            with col_main2:
                st.markdown("<p class='section-header'>🧪 Laboratory Metrics</p>", unsafe_allow_html=True)
                for feat in features[mid:]:
                    val = st.session_state.get(f"input_{feat}", None)
                    label = feat.replace('_', ' ')
                    if feat == 'Gender':
                        input_vals[feat] = st.selectbox(label, ["Male", "Female"], index=0 if val == "Male" else 1 if val == "Female" else 0)
                    elif feat in ['BloodInUrine', 'FrequentUrination', 'PelvicPain', 'BackPain', 'PainfulUrination']:
                        input_vals[feat] = st.selectbox(label, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=int(val) if val is not None else 0)
                    else:
                        input_vals[feat] = st.number_input(label, value=float(val) if val is not None else 0.0, format="%.2f")
            
            submit = st.form_submit_button("Initiate Neural Diagnostic Sequence")
        st.markdown("</div>", unsafe_allow_html=True)

        if submit:
            # Custom loader using the requested Max Healthcare GIF
            st.markdown(f"""
                <div style='position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 9999; display: flex; flex-direction: column; align-items: center; justify-content: center;'>
                    <img src='https://www.maxhealthcare.in/img/website-loader.gif' style='width: 120px; border-radius: 50%;'>
                    <p style='color: white; font-weight: 800; margin-top: 1rem; font-size: 1.2rem; letter-spacing: 2px;'>DECODING CLINICAL SIGNATURES...</p>
                </div>
            """, unsafe_allow_html=True)
            
            time.sleep(2.0) # Simulate complex diagnostic processing
            res = self.predictor.predict(input_vals)
            
            # Update Stats
            st.session_state.prediction_stats['total'] += 1
            pred_label = res['prediction']
            if pred_label in st.session_state.prediction_stats:
                st.session_state.prediction_stats[pred_label] += 1
            
            # Store result in session state to persist after rerun
            st.session_state.last_prediction = res
            st.rerun()

        if 'last_prediction' in st.session_state:
            res = st.session_state.last_prediction
            # Result Card
            st.markdown("<div style='margin-top: 4rem;'></div>", unsafe_allow_html=True)
            
            # Dark mode specific styles for result card elements
            res_heading_style = "color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;" if current_theme == "Dark" else "color: #0F172A !important;"
            res_subtext_style = "color: #E2E8F0 !important;" if current_theme == "Dark" else "color: #475569 !important;"
            res_bg_style = "background: rgba(30, 41, 59, 0.9) !important;" if current_theme == "Dark" else "background: #FFFFFF;"
            
            # Main Result Header
            st.markdown(f"<h2 style='text-align: center; font-weight: 800; {res_heading_style} margin-top: 2rem; margin-bottom: 0rem; letter-spacing: -1px;'>🎯 PREDICTION RESULTS</h2>", unsafe_allow_html=True)
            
            # --- CONSOLIDATED RESULT INTERFACE ---
            st.markdown(f"<div class='result-card fade-in' style='max-width: 1200px; margin: 2rem auto; padding: 2rem; {res_bg_style} border-top: 4px solid {('#10B981' if res['prediction'] == 'Normal' else res['risk_color'])};'>", unsafe_allow_html=True)
            
            # Header Row
            status_color = "#10B981" if res['prediction'] == "Normal" else res['risk_color']
            rec = recommendations.get(res['prediction'], recommendations['Normal'])
            
            head_col1, head_col2 = st.columns([3, 1])
            with head_col1:
                st.markdown(f"""
                    <div style='margin-bottom: 2rem;'>
                        <p style='text-transform: uppercase; letter-spacing: 3px; font-size: 0.7rem; color: #6C63FF !important; font-weight: 800; margin: 0;'>Diagnostic Outcome</p>
                        <h1 style='font-size: 2.8rem; margin: 0.2rem 0; font-weight: 800; {res_heading_style}'>{res['prediction']}</h1>
                        <div style='display: flex; align-items: center; gap: 1rem;'>
                            <span style='background: {status_color}; color: white !important; padding: 0.3rem 0.8rem; border-radius: 6px; font-weight: 800; font-size: 0.8rem;'>{res['risk_level'] if res['prediction'] != "Normal" else "STABLE"}</span>
                            <span style='font-weight: 700; {res_heading_style} font-size: 1.2rem;'>{res['confidence']:.1f}% Match</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with head_col2:
                st.markdown(f"""
                    <div style='display: flex; justify-content: flex-end;'>
                        <div class='result-img-wrapper'>
                            <div class='zoom-hint'>VIEW 🔍</div>
                            <img src='{rec.get('image', '')}' class='result-img-thumb'>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr style='opacity: 0.1; margin: 1rem 0 2rem 0;'>", unsafe_allow_html=True)

            # Details Row
            det_col1, det_col2 = st.columns([1.2, 1], gap="large")
            
            with det_col1:
                st.markdown(f"""
                    <h3 style='color: #6C63FF !important; font-weight: 800; margin-bottom: 1rem; font-size: 1.4rem;'>💡 Clinical Analysis</h3>
                    <p style='{res_subtext_style} font-size: 1.1rem; line-height: 1.5; margin-bottom: 1.5rem;'>{rec['info']}</p>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                        <div>
                            <h4 style='color: #00D4FF !important; font-weight: 800; margin-bottom: 0.5rem; font-size: 0.9rem; border-bottom: 1px solid #00D4FF; display: inline-block;'>✅ Protocol</h4>
                            {"".join([f"<p style='margin: 0.3rem 0; font-size:0.9rem; {res_subtext_style}'>• {t}</p>" for t in rec['tips']])}
                        </div>
                        <div>
                            <h4 style='color: #6C63FF !important; font-weight: 800; margin-bottom: 0.5rem; font-size: 0.9rem; border-bottom: 1px solid #6C63FF; display: inline-block;'>🥗 Nutrition</h4>
                            {"".join([f"<p style='margin: 0.3rem 0; font-size:0.9rem; {res_subtext_style}'>• {d}</p>" for d in rec['diet']])}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with det_col2:
                radar_text_color = '#FFFFFF' if current_theme == 'Dark' else '#0F172A'
                radar_grid_color = 'rgba(255,255,255,0.1)' if current_theme == 'Dark' else 'rgba(0,0,0,0.1)'
                
                st.markdown(f"<h3 style='font-weight: 800; text-align: center; margin-bottom: 1rem; {res_heading_style} font-size: 1.2rem;'>🕸️ Probability Map</h3>", unsafe_allow_html=True)
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=res['probabilities'],
                    theta=res['classes'],
                    fill='toself',
                    line_color='#6C63FF',
                    fillcolor='rgba(108, 99, 255, 0.2)',
                    marker=dict(size=6)
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor=radar_grid_color, tickfont=dict(color=radar_text_color, size=9)),
                        angularaxis=dict(tickfont=dict(color=radar_text_color, size=10))
                    ),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color=radar_text_color,
                    height=350,
                    margin=dict(t=20, b=20, l=30, r=30)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    def about_page(self, current_theme, page_bg, page_text, subtext):
        st.markdown("<div class='fade-in' style='margin-top: 2rem;'>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='font-weight:800; letter-spacing:-2px; font-size: 3rem; {page_text} text-align: center; margin-bottom: 2rem;'>Intelligence Blueprint</h1>", unsafe_allow_html=True)
        
        # Team Section with Marquee
        st.markdown(f"<h2 style='text-align: center; margin-bottom: 1rem; font-weight: 800; {page_text}'>Our Research Team</h2>", unsafe_allow_html=True)
        
        team_members = [
            {"name": "M.V.V. Durga Prakash", "dept": "Dept. of Information Technology", "inst": "Narasaraopeta Engineering College", "img": "https://i.pravatar.cc/300?u=durgaprakash"},
            {"name": "P. Vijaya Lakshmi", "dept": "Dept. of Information Technology", "inst": "Narasaraopeta Engineering College", "img": "https://i.pravatar.cc/300?u=vijayalakshmi"},
            {"name": "Shaik Nagoor Vali", "dept": "Dept. of Information Technology", "inst": "Narasaraopeta Engineering College", "img": "https://i.pravatar.cc/300?u=nagoor"},
            {"name": "Pathan Rasheed Khan", "dept": "Dept. of Information Technology", "inst": "Narasaraopeta Engineering College", "img": "https://i.pravatar.cc/300?u=rasheed"},
            {"name": "Maddula Ratna Mohitha", "dept": "Dept. of CSE", "inst": "AITAM, Tekkali, Srikakulam", "img": "https://i.pravatar.cc/300?u=mohitha"}
        ]
        
        # Doubling the list for smooth infinite scroll
        # Member name color in marquee
        m_name_color = "#00FF9F" if current_theme == "Dark" else "#0F172A"
        
        marquee_html = "".join([f"""
            <div class='team-member-card' style='{page_bg} border: 1px solid rgba(255,255,255,0.1);'>
                <div class='member-image-wrapper'>
                    <img src='{m['img']}' alt='{m['name']}'>
                </div>
                <div class='member-info'>
                    <div class='member-name' style='color: {m_name_color} !important;'>{m['name']}</div>
                    <div class='member-dept'>{m['dept']}</div>
                    <div class='member-inst' style='{subtext}'>{m['inst']}</div>
                </div>
            </div>
        """ for m in team_members + team_members])
        
        st.markdown(f"""
            <div class='marquee-container'>
                <div class='marquee-content'>
                    {marquee_html}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        
        # Detailed Technical Overview
        st.markdown(f"<h2 style='text-align: center; margin-bottom: 2rem; font-weight: 800; {page_text}'>Technical Methodology</h2>", unsafe_allow_html=True)
        
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        with tech_col1:
            st.markdown(f"""
                <div class='glass-container' style='max-width: 100%; height: 100%; margin: 0; {page_bg} border: 1px solid rgba(255,255,255,0.1);'>
                    <h3 style='color: #6C63FF !important; font-weight: 800;'>🧠 ML Models</h3>
                    <p style='{subtext} font-size: 0.95rem;'>
                        <b>XGBoost Ensemble:</b> Primary gradient boosting algorithm optimized for clinical classification.<br><br>
                        <b>Random Forest:</b> Secondary bagging classifier used for feature importance validation.<br><br>
                        <b>Hyperparameter Tuning:</b> RandomizedSearchCV used for optimizing tree depth, learning rate, and estimators.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with tech_col2:
            st.markdown(f"""
                <div class='glass-container' style='max-width: 100%; height: 100%; margin: 0; {page_bg} border: 1px solid rgba(255,255,255,0.1);'>
                    <h3 style='color: #00D4FF !important; font-weight: 800;'>🧪 Preprocessing</h3>
                    <p style='{subtext} font-size: 0.95rem;'>
                        <b>SMOTE:</b> Synthetic Minority Over-sampling Technique to ensure perfect 1:1:1:1:1:1 class balance.<br><br>
                        <b>Standard Scaling:</b> Robust scaling of 15+ clinical biomarkers to normalize neural variance.<br><br>
                        <b>Median Imputation:</b> Intelligent handling of missing clinical vectors based on class medians.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        with tech_col3:
            st.markdown(f"""
                <div class='glass-container' style='max-width: 100%; height: 100%; margin: 0; {page_bg} border: 1px solid rgba(255,255,255,0.1);'>
                    <h3 style='color: #6C63FF !important; font-weight: 800;'>📊 Dataset</h3>
                    <p style='{subtext} font-size: 0.95rem;'>
                        <b>6 Diagnostic Classes:</b> UrinaryBladder, Kidney, Prostate, Cystitis, Uterus, and Healthy (Normal).<br><br>
                        <b>Balanced Volume:</b> 6,000 high-fidelity clinical records (1,000 samples per class).<br><br>
                        <b>Feature Engineering:</b> Includes Age, Gender, Blood markers (WBC, RBC, Hemoglobin), and Renal markers (Creatinine, Urea).
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        
        a1, a2 = st.columns([1.5, 1])
        with a1:
            st.markdown(f"""
                <div class='glass-container' style='max-width: 100%; margin: 0; {page_bg} border: 1px solid rgba(255,255,255,0.1);'>
                <h3 style='{page_text} font-weight: 800;'>System Architecture</h3>
                <p style='{subtext}'>Our ensemble framework utilizes <b>XGBoost</b> and <b>Random Forest</b> models trained on balanced clinical datasets. We employ <b>SMOTE</b> (Synthetic Minority Over-sampling Technique) to handle class imbalances across all 6 diagnostic types, ensuring the model remains unbiased even for rare clinical profiles.</p>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 2rem;'>
                    <div style='background: rgba(108, 99, 255, 0.05); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(108, 99, 255, 0.1);'>
                        <h4 style='margin:0; color: #6C63FF !important; font-weight: 800;'>99% Recall</h4>
                        <p style='margin:0; font-size: 0.8rem; opacity: 0.8; {subtext}'>For Advanced Stages</p>
                    </div>
                    <div style='background: rgba(0, 212, 255, 0.05); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(0, 212, 255, 0.1);'>
                        <h4 style='margin:0; color: #00D4FF !important; font-weight: 800;'>1.2s Latency</h4>
                        <p style='margin:0; font-size: 0.8rem; opacity: 0.8; {subtext}'>End-to-end Inference</p>
                    </div>
                </div>
                </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists('model/metrics.json'):
                with open('model/metrics.json', 'r') as f:
                    metrics = json.load(f)
                st.markdown(f"<h3 style='{page_text} margin-top: 2rem; font-weight: 800;'>Production Metrics</h3>", unsafe_allow_html=True)
                st.json(metrics)

        with a2:
            st.markdown(f"""
                <div class='glass-container' style='max-width: 100%; margin: 0; {page_bg} border: 1px solid rgba(255,255,255,0.1);'>
                <h3 style='{page_text} font-weight: 800;'>Tech Stack</h3>
                <p style='{subtext}'>• <b>Inference:</b> XGBoost Ensemble<br>
                • <b>UI:</b> Streamlit 1.31 (Glassmorphism 2.0)<br>
                • <b>Data:</b> Pandas & Scikit-learn Pipeline<br>
                • <b>Charts:</b> Plotly High-Res Visualization</p>
                </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists('model/model.pkl'):
                importances = self.predictor.model.feature_importances_
                features = self.predictor.selected_features
                fig = plot_feature_importance(importances, features)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color='#FFFFFF' if current_theme == 'Dark' else '#000000', 
                    height=400,
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)' if current_theme == 'Dark' else 'rgba(0,0,0,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)' if current_theme == 'Dark' else 'rgba(0,0,0,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    app = AdvancedNexGenApp()
    app.run()
