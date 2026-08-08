import streamlit as st
import streamlit.components.v1 as components
import time
import mysql.connector
from mysql.connector import Error
import pandas as pd
from pypdf import PdfReader
from ai_engine import extract_skills
from graph_engine import run_cv_jd_pipeline

st.set_page_config(
    page_title="IntervAI Pro | AI Interview Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if st.session_state.dark_mode:
    _bg_base = "#030014"
    _bg_gradient = "radial-gradient(circle at 50% -20%, #1e1145, #030014 60%)"
    _text_color = "#F3F4F6"
    _card_bg = "rgba(255, 255, 255, 0.02)"
    _card_border = "rgba(255, 255, 255, 0.08)"
    _card_shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.37)"
    _sidebar_bg = "rgba(3, 0, 20, 0.9)"
    _sidebar_border = "rgba(255, 255, 255, 0.05)"
    _gradient_text = "linear-gradient(135deg, #FFFFFF 30%, #A78BFA 100%)"
    _badge_bg = "rgba(124, 58, 237, 0.15)"
    _badge_color = "#C084FC"
else:
    _bg_base = "#F7F5FC"
    _bg_gradient = "radial-gradient(circle at 50% -20%, #EDE9FE, #F7F5FC 60%)"
    _text_color = "#1F2937"
    _card_bg = "rgba(17, 24, 39, 0.03)"
    _card_border = "rgba(17, 24, 39, 0.08)"
    _card_shadow = "0 8px 32px 0 rgba(109, 40, 217, 0.08)"
    _sidebar_bg = "rgba(255, 255, 255, 0.85)"
    _sidebar_border = "rgba(17, 24, 39, 0.06)"
    _gradient_text = "linear-gradient(135deg, #1F2937 30%, #7C3AED 100%)"
    _badge_bg = "rgba(124, 58, 237, 0.08)"
    _badge_color = "#6D28D9"

# Injected custom CSS - includes a permanent fix for the sidebar toggle button disappearing
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: {_bg_base} !important;
        color: {_text_color};
    }}

    .stApp {{
        background: {_bg_gradient};
    }}

    /* === 🛠️ Definitive fix for the sidebar toggle button === */
    button[data-testid="stSidebarCollapseButton"] {{
        background: linear-gradient(135deg, #6D28D9 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        opacity: 1 !important;
        visibility: visible !important;
        z-index: 999999 !important;
        box-shadow: 0 0 15px rgba(109, 40, 217, 0.6) !important;
        position: fixed !important;
        top: 15px !important;
    }}

    button[data-testid="stSidebarCollapseButton"]:hover {{
        transform: scale(1.05) !important;
        box-shadow: 0 0 25px rgba(109, 40, 217, 0.9) !important;
    }}

    header {{
        background: transparent !important;
    }}

    /* Premium glass-morphism effects for cards */
    .premium-card {{
        background: {_card_bg};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {_card_border};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: {_card_shadow};
        transition: all 0.3s ease;
    }}

    .premium-card:hover {{
        border-color: rgba(124, 58, 237, 0.4);
        box-shadow: 0 8px 40px 0 rgba(124, 58, 237, 0.1);
    }}

    .gradient-text {{
        background: {_gradient_text};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -1px;
    }}

    .stButton>button {{
        background: linear-gradient(90deg, #6D28D9 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        width: 100%;
    }}

    [data-testid="stSidebar"] {{
        background-color: {_sidebar_bg} !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid {_sidebar_border};
    }}

    .badge {{
        background: {_badge_bg};
        color: {_badge_color};
        border: 1px solid rgba(124, 58, 237, 0.3);
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.85rem;
        display: inline-block;
    }}
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    /* ---- keyframes ---- */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes softGlow {
        0%, 100% { box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35); }
        50%      { box-shadow: 0 6px 34px rgba(99, 102, 241, 0.75); }
    }
    @keyframes badgePulse {
        0%, 100% { transform: scale(1);    box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.45); }
        50%      { transform: scale(1.08); box-shadow: 0 0 0 10px rgba(124, 58, 237, 0); }
    }
    @keyframes shimmer {
        0%   { background-position: -300px 0; }
        100% { background-position: 300px 0; }
    }

    /* ---- entrance: every block Streamlit renders fades/slides in ----
       The modern data-testid is listed first; the legacy .main .block-container
       is kept only as a fallback for older Streamlit builds (it matches
       nothing on current versions). */
    [data-testid="stAppViewContainer"] [data-testid="stMainBlockContainer"] > div,
    [data-testid="stAppViewContainer"] .main .block-container > div {
        animation: fadeSlideUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    [data-testid="stSidebar"] {
        animation: fadeIn 0.6s ease both;
    }

    /* ---- premium-card: lift + cursor-tracked spotlight ---- */
    .premium-card {
        animation: fadeSlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        position: relative;
        overflow: hidden;
    }
    /* the spotlight follows the cursor via --mx/--my set in the JS below */
    .premium-card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: radial-gradient(
            520px circle at var(--mx, 50%) var(--my, 50%),
            rgba(124, 58, 237, 0.22),
            transparent 45%
        );
        opacity: 0;
        transition: opacity 0.35s ease;
        pointer-events: none;
        z-index: 0;
    }
    .premium-card:hover::before { opacity: 1; }
    .premium-card > * { position: relative; z-index: 1; }
    .premium-card:hover {
        transform: translateY(-6px) scale(1.008);
        border-color: rgba(124, 58, 237, 0.55) !important;
        box-shadow: 0 14px 44px rgba(124, 58, 237, 0.22) !important;
    }

    /* ---- gradient headline text: living shimmer (wider travel, faster) ---- */
    .gradient-text {
        background-size: 300% 300%;
        animation: gradientShift 4s ease infinite;
    }

    /* ---- badge: breathing pulse with an expanding halo ---- */
    .badge {
        animation: badgePulse 2.2s ease-in-out infinite;
    }

    /* ---- buttons: lift, glow, and a light sheen sweeping across on hover ---- */
    .stButton > button {
        position: relative;
        overflow: hidden;
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease !important;
    }
    .stButton > button::after {
        content: "";
        position: absolute;
        top: 0;
        left: -60%;
        width: 40%;
        height: 100%;
        background: linear-gradient(
            120deg,
            transparent,
            rgba(255, 255, 255, 0.38),
            transparent
        );
        transform: skewX(-20deg);
        transition: left 0.55s ease;
        pointer-events: none;
    }
    .stButton > button:hover::after { left: 130%; }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        animation: softGlow 1.5s ease-in-out infinite;
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
    }

    /* ---- cursor-tracked ambient glow (element injected by the JS below) ---- */
    #cursor-glow {
        position: fixed;
        top: 0;
        left: 0;
        width: 460px;
        height: 460px;
        margin-left: -230px;
        margin-top: -230px;
        border-radius: 50%;
        background: radial-gradient(
            circle,
            rgba(124, 58, 237, 0.20) 0%,
            rgba(79, 70, 229, 0.10) 35%,
            transparent 70%
        );
        pointer-events: none;
        z-index: 0;
        opacity: 0;
        transition: opacity 0.4s ease;
        will-change: transform;
    }
    #cursor-glow.visible { opacity: 1; }

    /* ---- sidebar nav radio options: smooth hover shift ---- */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        transition: transform 0.2s ease, opacity 0.2s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        transform: translateX(-4px);
        opacity: 0.85;
    }

    /* ---- expanders: gentle open/close feel ---- */
    [data-testid="stExpander"] {
        transition: box-shadow 0.3s ease;
    }
    [data-testid="stExpander"]:hover {
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.08);
    }

    /* ---- text inputs / text areas: focus glow instead of a hard snap ---- */
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
        transition: box-shadow 0.25s ease, border-color 0.25s ease !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.25) !important;
    }

    /* ---- progress bar fill: subtle shimmer while it moves ---- */
    [data-testid="stProgress"] > div > div > div {
        background-image: linear-gradient(
            90deg,
            rgba(255,255,255,0) 0%,
            rgba(255,255,255,0.35) 50%,
            rgba(255,255,255,0) 100%
        ), linear-gradient(90deg, #6D28D9, #4F46E5) !important;
        background-size: 200px 100%, 100% 100%;
        animation: shimmer 1.4s linear infinite;
    }

    /* ---- metric-style small cards (m1..m4 in dashboard) get a tiny hover lift too ---- */
    .premium-card h2 {
        transition: transform 0.2s ease;
    }
    .premium-card:hover h2 {
        transform: scale(1.03);
    }

    /* ---- respect users who prefer reduced motion ---- */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
        }
    }
    </style>
""", unsafe_allow_html=True)


components.html("""
<script>
(function () {
    const doc = window.parent.document;
    if (doc.getElementById("cursor-glow")) { return; }   // already installed

    // Ambient glow that trails the cursor across the whole page.
    const glow = doc.createElement("div");
    glow.id = "cursor-glow";
    doc.body.appendChild(glow);

    let targetX = 0, targetY = 0, currentX = 0, currentY = 0, running = false;

    function animate() {
        // Ease toward the cursor so the glow trails rather than snaps.
        currentX += (targetX - currentX) * 0.12;
        currentY += (targetY - currentY) * 0.12;
        glow.style.transform = `translate(${currentX}px, ${currentY}px)`;
        if (Math.abs(targetX - currentX) > 0.5 || Math.abs(targetY - currentY) > 0.5) {
            requestAnimationFrame(animate);
        } else {
            running = false;
        }
    }

    doc.addEventListener("mousemove", function (e) {
        targetX = e.clientX;
        targetY = e.clientY;
        glow.classList.add("visible");
        if (!running) { running = true; requestAnimationFrame(animate); }

        // Per-card spotlight: feed the cursor position to the hovered card.
        const card = e.target.closest ? e.target.closest(".premium-card") : null;
        if (card) {
            const r = card.getBoundingClientRect();
            card.style.setProperty("--mx", (e.clientX - r.left) + "px");
            card.style.setProperty("--my", (e.clientY - r.top) + "px");
        }
    }, { passive: true });

    doc.addEventListener("mouseleave", function () {
        glow.classList.remove("visible");
    });
})();
</script>
""", height=0)

def get_db_connection():
    try:
        return mysql.connector.connect(host='localhost', database='intervai_db', user='root', password='')
    except Error:
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interviews (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                job_title VARCHAR(255) NOT NULL,
                experience_level VARCHAR(100),
                job_description TEXT,
                score INT DEFAULT 0,
                ai_evaluation TEXT,
                session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_qa (
                id INT AUTO_INCREMENT PRIMARY KEY,
                interview_id INT,
                question_text TEXT NOT NULL,
                user_answer TEXT,
                FOREIGN KEY (interview_id) REFERENCES interviews(id) ON DELETE CASCADE
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()

# Auto-create/repair tables on startup
init_db()

def create_user(name, email, password):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)', (name, email, password))
            conn.commit()
            return True
        except mysql.connector.IntegrityError:
            return False
        finally:
            cursor.close()
            conn.close()
    return False

def authenticate_user(email, password):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, name FROM users WHERE email=%s AND password=%s', (email, password))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result: return result[0], result[1]
    return None

def create_interview_session(user_id, job_title, exp_level, job_desc):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO interviews (user_id, job_title, experience_level, job_description) VALUES (%s, %s, %s, %s)', (user_id, job_title, exp_level, job_desc))
            conn.commit()
            return cursor.lastrowid
        except Error:
            return None
        finally:
            cursor.close()
            conn.close()
    return None

def save_question_answer(interview_id, question_text, user_answer):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO interview_qa (interview_id, question_text, user_answer) VALUES (%s, %s, %s)', (interview_id, question_text, user_answer))
        conn.commit()
        cursor.close()
        conn.close()

def finalize_interview(interview_id, score, evaluation_text):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE interviews SET score = %s, ai_evaluation = %s WHERE id = %s', (score, evaluation_text, interview_id))
        conn.commit()
        cursor.close()
        conn.close()

def get_user_interview_history(user_id):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT id, job_title, experience_level, score, ai_evaluation, session_date FROM interviews WHERE user_id=%s ORDER BY session_date ASC', (user_id,))
            return cursor.fetchall()
        except Error:
            return []
        finally:
            cursor.close()
            conn.close()
    return []


if "page" not in st.session_state: st.session_state.page = "Landing"
if "is_logged_in" not in st.session_state: st.session_state.is_logged_in = False
if "user_id" not in st.session_state: st.session_state.user_id = None
if "user_name" not in st.session_state: st.session_state.user_name = ""
if "cv_uploaded" not in st.session_state: st.session_state.cv_uploaded = False
if "cv_skills" not in st.session_state: st.session_state.cv_skills = {}
if "skill_gap" not in st.session_state: st.session_state.skill_gap = {}
if "generated_questions" not in st.session_state: st.session_state.generated_questions = []
if "interview_answers" not in st.session_state: st.session_state.interview_answers = []
if "cv_file_name" not in st.session_state: st.session_state.cv_file_name = None
if "interview_started" not in st.session_state: st.session_state.interview_started = False
if "current_question" not in st.session_state: st.session_state.current_question = 0
if "current_interview_id" not in st.session_state: st.session_state.current_interview_id = None
if "mock_questions" not in st.session_state:
    st.session_state.mock_questions = [
        "Tell me about a challenging technical project you worked on, specifically focusing on hardware-software integration.",
        "How do you approach debugging a memory allocation or latency issue in an embedded or real-time system?",
        "Why do you want to join our company, and how do your skills align with the Job Description provided?"
    ]

def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()


def render_landing():
    st.markdown("<div style='text-align: center; padding: 60px 0 40px 0;'>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>✨ Next-Gen AI Mock Interviews</span>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 4rem; margin-top: 15px;' class='gradient-text'>Ace your next job interview <br>with the power of AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.25rem; color: #9CA3AF; max-width: 700px; margin: 20px auto;'>Upload your CV, enter the job description, and let the AI interview engine test you and give you a detailed evaluation that guarantees you're ready.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("Start training for free now", key="hero_cta"):
            navigate_to("Auth")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 40px 0;'>", unsafe_allow_html=True)

    # Features
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px;' class='gradient-text'>Platform Features</h2>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown('<div class="premium-card"><h3 style="color: #A78BFA;">📄 CV Analysis</h3><p style="color: #9CA3AF;">An LLM-powered processing system that analyzes the strengths, weaknesses, and experience in your CV.</p></div>', unsafe_allow_html=True)
    with f_col2:
        st.markdown('<div class="premium-card"><h3 style="color: #60A5FA;">🎯 Smart Live Simulation</h3><p style="color: #9CA3AF;">Dynamically generates technical and behavioral questions tailored to the company and job description.</p></div>', unsafe_allow_html=True)
    with f_col3:
        st.markdown('<div class="premium-card"><h3 style="color: #34D399;">📊 Scoring & Evaluation</h3><p style="color: #9CA3AF;">Get a complete dashboard showing your strengths and precise mistakes, along with professional alternative phrasing.</p></div>', unsafe_allow_html=True)

    # Pricing
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 40px 0;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px;' class='gradient-text'>Flexible Subscription Plans</h2>", unsafe_allow_html=True)

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown('<div class="premium-card" style="border-top: 3px solid #9CA3AF;"><h3>Free Plan</h3><h2 style="margin: 15px 0;">$0 <span style="font-size: 1rem; color: #9CA3AF;">/ month</span></h2><p style="color: #9CA3AF;">• One full interview per month<br>• Basic CV analysis<br>• Simplified report format</p></div>', unsafe_allow_html=True)
        if st.button("Choose the Free Plan", key="btn_free"): navigate_to("Auth")

    with p_col2:
        st.markdown('<div class="premium-card" style="border-top: 3px solid #7C3AED; background: rgba(124, 58, 237, 0.03);"><h3>Pro Plan 🔥</h3><h2 style="margin: 15px 0;">$29 <span style="font-size: 1rem; color: #9CA3AF;">/ month</span></h2><p style="color: #A78BFA;">• Unlimited AI-powered interviews<br>• Advanced analysis & JD matching<br>• Precise AI answer evaluation</p></div>', unsafe_allow_html=True)
        if st.button("Subscribe to Pro Now", key="btn_pro"): navigate_to("Auth")

    with p_col3:
        st.markdown('<div class="premium-card" style="border-top: 3px solid #06B6D4;"><h3>Premium Plan 👑</h3><h2 style="margin: 15px 0;">$79 <span style="font-size: 1rem; color: #9CA3AF;">/ month</span></h2><p style="color: #9CA3AF;">• All Pro plan features<br>• Live voice interview simulation<br>• Custom reports, shareable with companies</p></div>', unsafe_allow_html=True)
        if st.button("Subscribe to Premium Now", key="btn_premium"): navigate_to("Auth")

# --- PAGE: AUTHENTICATION (CONNECTED TO MYSQL) ---
def render_auth():
    st.markdown("<div style='max-width: 450px; margin: 40px auto;'>", unsafe_allow_html=True)
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;' class='gradient-text'>Welcome to IntervAI Pro</h2>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Create New Account"])
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In", key="btn_login_submit"):
            res = authenticate_user(email, password)
            if res:
                st.session_state.is_logged_in = True
                st.session_state.user_id, st.session_state.user_name = res[0], res[1]
                st.toast("Logged in successfully! 🎉")
                navigate_to("Dashboard")
            else:
                st.error("Invalid login credentials, or the server is not connected.")

    with tab2:
        new_name = st.text_input("Full Name", key="reg_name")
        new_email = st.text_input("Email", key="reg_email")
        new_password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Create Account", key="btn_reg_submit"):
            if new_name and new_email and new_password:
                if create_user(new_name, new_email, new_password):
                    st.success("Account created successfully! Switch to the Login tab now.")
                else:
                    st.error("This email is already registered.")
            else:
                st.error("Please fill in all the fields.")

    st.markdown("<div style='text-align: center; margin-top: 15px; color:#9CA3AF;'>Or continue with a global account</div>", unsafe_allow_html=True)
    st.button("🌐 Central Authorization (Google)")
    st.markdown("</div></div>", unsafe_allow_html=True)

# --- PAGE: USER DASHBOARD (DYNAMIC WITH MYSQL) ---
def render_dashboard():
    st.markdown(f"<h1 class='gradient-text'>Strategic Dashboard | Welcome, {st.session_state.user_name} 👋</h1>", unsafe_allow_html=True)

    history = get_user_interview_history(st.session_state.user_id)
    total_interviews = len(history)
    valid_scores = [item['score'] for item in history if item.get('score') is not None]
    avg_score = int(sum(valid_scores) / len(valid_scores)) if len(valid_scores) > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f"<div class='premium-card'><small style='color: #9CA3AF;'>Total Interviews</small><h2 style='color:#7C3AED;'>{total_interviews}</h2></div>", unsafe_allow_html=True)
    with m2: st.markdown(f"<div class='premium-card'><small style='color: #9CA3AF;'>Average Score</small><h2 style='color:#06B6D4;'>{avg_score}%</h2></div>", unsafe_allow_html=True)
    with m3: st.markdown(f"<div class='premium-card'><small style='color: #9CA3AF;'>CV Status</small><h2 style='color:#34D399;'>{'Analyzed & Uploaded' if st.session_state.cv_uploaded else 'Not uploaded yet'}</h2></div>", unsafe_allow_html=True)
    with m4: st.markdown("<div class='premium-card'><small style='color: #9CA3AF;'>Current Plan</small><h2 style='color:#F59E0B;'>Pro Plan</h2></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("<div class='premium-card'><h3>📈 Technical Performance Progress Curve</h3>", unsafe_allow_html=True)
        if total_interviews > 0:
            df = pd.DataFrame(history)
            df['session_date'] = pd.to_datetime(df['session_date']).dt.strftime('%m/%d %H:%M')
            st.line_chart(data=df, x='session_date', y='score', use_container_width=True)
            for index, row in df.iterrows():
                with st.expander(f"💼 {row.get('job_title')} - Score: {row.get('score')}%"):
                    st.info(row.get('ai_evaluation') or "No text report available.")
        else:
            st.info("You haven't done any interviews yet to generate the chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="premium-card"><h3>🤖 Personalized AI Recommendations</h3><p style="color: #9CA3AF; font-size: 0.9rem; line-height: 1.6;">• Focus on explaining <b>System Architecture</b> when asked about your past projects.<br><br>• Your English is excellent, but try to reduce hesitation tokens.<br><br>• Your technical skills are very strong and match 4 currently open positions.</p></div>', unsafe_allow_html=True)

def _extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# --- PAGE: UPLOAD SYSTEM (CONNECTED TO MYSQL) ---
def render_upload():
    st.markdown("<h1 class='gradient-text'>Cloud Data Upload & Analysis System</h1>", unsafe_allow_html=True)

    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("1. Upload Your CV")
    uploaded_file = st.file_uploader("Choose your CV file (PDF format)", type=["pdf"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.cv_file_name:
        with st.spinner("Analyzing your CV with AI..."):
            cv_text = _extract_pdf_text(uploaded_file)
            result = extract_skills(cv_text, document_type="CV")

        if not cv_text.strip():
            st.session_state.cv_uploaded = False
            st.session_state.cv_skills = {}
            st.error("Could not extract any text from the file. Make sure the PDF contains real text, not a scanned image.")
        elif "error" in result:
            st.session_state.cv_uploaded = False
            st.session_state.cv_skills = {}
            st.error(f"Could not analyze the CV: {result['error']}")
        else:
            total = len(result["technical_skills"]) + len(result["soft_skills"]) + len(result["languages"])
            st.session_state.cv_uploaded = True
            st.session_state.cv_skills = result
            st.session_state.cv_file_name = uploaded_file.name
            st.success(f"CV analyzed successfully — {total} skills extracted.")

    if st.session_state.cv_skills:
        tech = st.session_state.cv_skills.get("technical_skills", [])
        soft = st.session_state.cv_skills.get("soft_skills", [])
        langs = st.session_state.cv_skills.get("languages", [])
        with st.expander(f"View extracted skills ({len(tech) + len(soft) + len(langs)})", expanded=True):
            if tech:
                st.markdown("**Technical Skills**")
                for skill in tech:
                    st.markdown(f"- {skill}")
            if soft:
                st.markdown("**Soft Skills**")
                for skill in soft:
                    st.markdown(f"- {skill}")
            if langs:
                st.markdown("**Languages**")
                for lang in langs:
                    st.markdown(f"- {lang}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("2. Target Job Details")
    job_title = st.text_input("Target Job Title", "Software Engineer")
    job_desc = st.text_area("Enter the Job Description", value="Write the job description details here...")

    col1, col2 = st.columns(2)
    with col1: experience = st.selectbox("Current Experience Level", ["Junior (0-2 years)", "Mid-Level (2-5 years)", "Senior (5+ years)"])
    with col2: lang_level = st.selectbox("Required Interview Language", ["English (Professional)", "Arabic / العربية"])

    if st.button("🔍 Analyze the Gap Between Your Skills and the Job"):
        if not st.session_state.cv_skills:
            st.warning("Upload your CV and analyze your skills first (step 1) before running the gap analysis.")
        elif not job_desc.strip() or job_desc.strip() == "Write the job description details here...":
            st.warning("Enter the actual job description first.")
        else:
            with st.spinner("Analyzing the job requirements and comparing them to your skills..."):
                pipeline_result = run_cv_jd_pipeline(jd_text=job_desc, cv_skills=st.session_state.cv_skills)
                if pipeline_result.get("error"):
                    st.session_state.skill_gap = {}
                    st.session_state.generated_questions = []
                    st.error(f"Could not analyze the job description: {pipeline_result['error']}")
                else:
                    st.session_state.skill_gap = pipeline_result["skill_gap"]
                    st.session_state.generated_questions = pipeline_result.get("questions", {}).get("questions", [])

    if st.session_state.skill_gap:
        gap = st.session_state.skill_gap
        missing_tech = gap.get("missing_technical_skills", [])
        missing_soft = gap.get("missing_soft_skills", [])
        missing_langs = gap.get("missing_languages", [])
        total_missing = len(missing_tech) + len(missing_soft) + len(missing_langs)
        with st.expander(f"Gap analysis result ({total_missing} missing skills)", expanded=True):
            if total_missing == 0:
                st.success("Your skills cover all the requirements visible in the job description! 🎉")
            else:
                if missing_tech:
                    st.markdown("**Required technical skills you're missing**")
                    for s in missing_tech:
                        st.markdown(f"- {s}")
                if missing_soft:
                    st.markdown("**Required soft skills you're missing**")
                    for s in missing_soft:
                        st.markdown(f"- {s}")
                if missing_langs:
                    st.markdown("**Required languages you're missing**")
                    for s in missing_langs:
                        st.markdown(f"- {s}")

    if st.session_state.generated_questions:
        st.info(
            f"🎯 {len(st.session_state.generated_questions)} gap-prioritized interview questions are ready. "
            "Start the session below to answer them in the Interview Simulator."
        )

    if st.button("Save Data & Generate the AI Interview Engine"):
        if not job_title.strip() or not job_desc.strip():
            st.warning("Please make sure all job detail fields are filled in.")
        elif not st.session_state.cv_skills:
            st.warning("Upload and analyze your CV first (step 1).")
        else:
            if not st.session_state.generated_questions:
                with st.spinner("Generating gap-prioritized interview questions..."):
                    pipeline_result = run_cv_jd_pipeline(jd_text=job_desc, cv_skills=st.session_state.cv_skills)
                    if pipeline_result.get("error"):
                        st.error(f"Could not generate the interview questions: {pipeline_result['error']}")
                        st.stop()
                    st.session_state.skill_gap = pipeline_result["skill_gap"]
                    st.session_state.generated_questions = pipeline_result.get("questions", {}).get("questions", [])

            with st.spinner("Initializing the AI interview server in the database..."):
                int_id = create_interview_session(st.session_state.user_id, job_title, experience, job_desc)
                if int_id:
                    st.session_state.current_interview_id = int_id
                    st.session_state.interview_started = True
                    st.session_state.current_question = 0
                    st.session_state.interview_answers = []
                    st.toast("Session and engine built successfully! 🚀")
                    navigate_to("Interview Engine")
                else:
                    st.error("Database connection error. Make sure XAMPP is running.")
    st.markdown("</div>", unsafe_allow_html=True)

def render_interview():
    st.markdown("<h1 class='gradient-text'>🤖 Live AI Interview Simulator</h1>", unsafe_allow_html=True)

    if not st.session_state.interview_started or st.session_state.current_interview_id is None:
        st.markdown("<div class='premium-card' style='text-align: center; padding: 40px;'>", unsafe_allow_html=True)
        st.write("The engine is waiting for data setup from the Upload System page.")
        if st.button("Go to the Upload page now"):
            navigate_to("Upload System")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Use the gap-prioritized questions generated by the LangGraph pipeline.
        # Fall back to the static sample questions only if generation produced nothing.
        generated = st.session_state.generated_questions
        if generated:
            question_texts = [q["question"] for q in generated]
        else:
            question_texts = st.session_state.mock_questions

        curr_idx = st.session_state.current_question
        total_qs = len(question_texts)

        if curr_idx < total_qs:
            current_q_text = question_texts[curr_idx]
           
            st.markdown(f"<div class='premium-card' style='border-left: 4px solid #7C3AED;'><h5>Current Question ({curr_idx + 1} of {total_qs}):</h5><h3 style='color: #A78BFA;'>{current_q_text}</h3></div>", unsafe_allow_html=True)
            user_ans = st.text_area("Enter your full answer here:", key=f"ans_{curr_idx}", height=120)

            def _record_answer():
                save_question_answer(st.session_state.current_interview_id, current_q_text, user_ans)
                st.session_state.interview_answers.append({"question": current_q_text, "answer": user_ans})

            if curr_idx < total_qs - 1:
                if st.button("Submit Answer & Go to Next Question ➡️"):
                    _record_answer()
                    st.session_state.current_question += 1
                    st.toast("Answer saved successfully!")
                    st.rerun()
            else:
                if st.button("Finish Interview & Get Results 🎓"):
                    _record_answer()
                    # Simulate saving the overall evaluation to the database
                    finalize_interview(st.session_state.current_interview_id, 87, "Excellent engineering performance — your answers are precise and show a deep understanding of embedded systems and system architecture.")
                    st.session_state.current_question += 1
                    st.rerun()
        else:
            st.markdown("<div class='premium-card' style='background: rgba(52, 211, 153, 0.03); border-color: #34D399;'><h3>🎉 Interview completed successfully! Generating your in-depth evaluation results...</h3></div>", unsafe_allow_html=True)
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.003)
                progress_bar.progress(percent_complete + 1)

            st.markdown("<h2 class='gradient-text'>Final Report & Technical Performance Analysis</h2>", unsafe_allow_html=True)
            score_col1, score_col2 = st.columns([1, 2])
            with score_col1:
                st.markdown("<div class='premium-card' style='text-align: center;'><h4>Overall Score</h4><h1 style='font-size: 4rem; color: #34D399;'>87%</h1><span class='badge'>Ready for the real interview</span></div>", unsafe_allow_html=True)
            with score_col2:
                st.markdown('<div class="premium-card"><h4>Skill Evaluation Breakdown:</h4><p><b>• Technical Answer Depth:</b> 9/10 - You showed excellent understanding of the engineering architecture and hardware/software integration.</p><p><b>• Structure (STAR Methodology):</b> 8/10 - Your answers were logically organized and methodical.</p></div>', unsafe_allow_html=True)

            if st.session_state.interview_answers:
                with st.expander(f"📝 Session transcript ({len(st.session_state.interview_answers)} answers saved)", expanded=False):
                    for i, item in enumerate(st.session_state.interview_answers, start=1):
                        st.markdown(f"**Q{i}. {item['question']}**")
                        st.markdown(f"<span style='color:#9CA3AF;'>{item['answer'] or '(no answer given)'}</span>", unsafe_allow_html=True)
                        st.markdown("---")

        
            if generated:
                passed = sum(1 for q in generated if q.get("passes_gate"))
                gap_qs = sum(1 for q in generated if q.get("is_gap_skill"))
                existing_qs = len(generated) - gap_qs
                with st.expander(
                    f"⚙️ Question-generation diagnostics — "
                    f"coverage: {gap_qs} gap / {existing_qs} existing &nbsp;·&nbsp; "
                    f"Answerability Gate: {passed}/{len(generated)} passed",
                    expanded=False,
                ):
                    for i, q in enumerate(generated, start=1):
                        gate_icon = "✅" if q.get("passes_gate") else "⚠️"
                        gap_label = "gap skill" if q.get("is_gap_skill") else "existing skill"
                        
                        entities = q.get("content_entities")
                        entities_label = f"{entities:.2f}" if entities is not None else "n/a (soft skill)"
                        st.markdown(
                            f"**Q{i}** — targets <b>{q['targets_skill']}</b> ({gap_label}) &nbsp;|&nbsp; "
                            f"AS = <b>{q['answerability_score']:.2f}</b> {gate_icon} "
                            f"<span style='color:#9CA3AF; font-size:0.85rem;'>"
                            f"(entities {entities_label} · clarity {q['context_clarity']:.2f} · specificity {q['task_specificity']:.2f})"
                            f"</span>",
                            unsafe_allow_html=True,
                        )

            if st.button("Return to Dashboard & Track the Curve"):
                st.session_state.interview_started = False
                st.session_state.current_question = 0
                st.session_state.current_interview_id = None
                st.session_state.interview_answers = []
                navigate_to("Dashboard")

def render_billing():
    st.markdown("<h1 class='gradient-text'>💳 Subscription Management & Global Billing System</h1>", unsafe_allow_html=True)
    st.markdown("<div class='premium-card'><h3>Current Subscription Details</h3><p>You are currently subscribed to: <b>Pro Plan</b></p><p>Next renewal date: <b>June 30, 2026</b></p><span class='badge'>Status: Active via Stripe</span></div>", unsafe_allow_html=True)


st.sidebar.markdown("<h2 class='gradient-text' style='text-align:center;'>IntervAI Pro 🤖</h2>", unsafe_allow_html=True)
st.sidebar.toggle("🌙 Dark Mode", key="dark_mode")
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)

if not st.session_state.is_logged_in:
    available_pages = ["Home (Landing Page)", "Login Portal"]
    default_idx = 0 if st.session_state.page == "Landing" else 1

    nav_selection = st.sidebar.radio("Main Menu", available_pages, index=default_idx)
    st.session_state.page = "Landing" if nav_selection == "Home (Landing Page)" else "Auth"
else:
    st.sidebar.markdown(f"<p style='text-align:center; color:#A78BFA;'>👤 {st.session_state.user_name}</p>", unsafe_allow_html=True)
    menu_options = {
        "📊 Dashboard": "Dashboard",
        "📂 Upload System": "Upload System",
        "🎙️ Interview Simulator (AI Engine)": "Interview Engine",
        "💳 Subscriptions & Billing": "Billing",
    }

    if st.session_state.page not in menu_options.values():
        st.session_state.page = "Dashboard"

    list_values = list(menu_options.values())
    list_keys = list(menu_options.keys())
    default_idx = list_values.index(st.session_state.page)

    selection = st.sidebar.radio("Cloud Navigation Panel", list_keys, index=default_idx)
    st.session_state.page = menu_options[selection]

    st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    if st.sidebar.button("Log Out"):
        st.session_state.is_logged_in = False
        st.session_state.user_id = None
        st.session_state.user_name = ""
        st.session_state.interview_started = False
        st.session_state.current_interview_id = None
        navigate_to("Landing")

if st.session_state.page == "Landing": render_landing()
elif st.session_state.page == "Auth": render_auth()
elif st.session_state.page == "Dashboard": render_dashboard()
elif st.session_state.page == "Upload System": render_upload()
elif st.session_state.page == "Interview Engine": render_interview()
elif st.session_state.page == "Billing": render_billing()