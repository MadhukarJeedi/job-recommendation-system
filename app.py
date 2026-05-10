import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS  –  Dark, neon-accent, editorial feel
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main container ── */
.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1200px;
}

/* ── Hero header ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #00f5d4 0%, #7b61ff 60%, #ff6b9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 300;
    color: #8888aa;
    letter-spacing: 0.02em;
    margin-bottom: 2.5rem;
}

/* ── Divider ── */
.neon-divider {
    height: 2px;
    background: linear-gradient(90deg, #00f5d4, #7b61ff, transparent);
    border: none;
    margin: 1.5rem 0 2rem 0;
    border-radius: 2px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #10101a;
    border-right: 1px solid #1e1e30;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem;
}
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #00f5d4;
    margin-bottom: 1rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.sidebar-hint {
    font-size: 0.8rem;
    color: #555577;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* ── Text area ── */
textarea {
    background: #13131f !important;
    border: 1px solid #2a2a40 !important;
    color: #e8e8f0 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
}
textarea:focus {
    border-color: #00f5d4 !important;
    box-shadow: 0 0 0 2px rgba(0,245,212,0.15) !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00f5d4, #7b61ff);
    color: #0a0a0f;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    margin-top: 1rem;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ── Sliders ── */
[data-testid="stSlider"] .thumb {
    background: #00f5d4;
}

/* ── Metric cards ── */
.metric-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00f5d4; }
.metric-num {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00f5d4, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.75rem;
    color: #666688;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Job cards ── */
.job-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    transition: border-color 0.25s, transform 0.2s;
    overflow: hidden;
}
.job-card:hover {
    border-color: #7b61ff;
    transform: translateY(-2px);
}
.job-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px;
    height: 100%;
    border-radius: 4px 0 0 4px;
}
.rank-1::before { background: linear-gradient(180deg, #00f5d4, #7b61ff); }
.rank-2::before { background: linear-gradient(180deg, #7b61ff, #ff6b9d); }
.rank-3::before { background: linear-gradient(180deg, #ff6b9d, #ffb347); }
.rank-4::before { background: linear-gradient(180deg, #ffb347, #00f5d4); }
.rank-5::before { background: linear-gradient(180deg, #00f5d4, #00adb5); }

.rank-badge {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555577;
    margin-bottom: 0.5rem;
}
.job-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.6rem;
    line-height: 1.3;
}
.score-badge {
    display: inline-block;
    background: rgba(0,245,212,0.12);
    color: #00f5d4;
    border: 1px solid rgba(0,245,212,0.25);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    margin-bottom: 0.8rem;
}
.skills-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555577;
    margin-bottom: 0.35rem;
}
.skill-tag {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    color: #9999bb;
    font-size: 0.75rem;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    margin: 0.15rem 0.15rem 0.15rem 0;
}
.apply-link {
    display: inline-block;
    margin-top: 1rem;
    color: #7b61ff;
    font-size: 0.85rem;
    font-weight: 500;
    text-decoration: none;
    border-bottom: 1px solid rgba(123,97,255,0.3);
    transition: color 0.2s;
}
.apply-link:hover { color: #00f5d4; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #333355;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #444466;
    margin-bottom: 0.5rem;
}

/* ── Skill chip suggestions ── */
.chip-container { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.chip {
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    color: #7b61ff;
    font-size: 0.75rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    cursor: pointer;
}

/* ── Section heading ── */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8e8f0;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div {
    background: #13131f !important;
    border: 1px solid #2a2a40 !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div { background: linear-gradient(90deg, #00f5d4, #7b61ff) !important; }

/* ── Info boxes ── */
.stAlert {
    background: #13131f !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA & MODEL  (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_and_train(uploaded_file=None):
    """Load CSV and fit TF-IDF model."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Demo data when no file is uploaded
        demo = {
            'jobtitle': [
                'Data Scientist', 'Machine Learning Engineer', 'Python Developer',
                'Data Analyst', 'AI Research Scientist', 'Backend Engineer',
                'NLP Engineer', 'Data Engineer', 'Business Intelligence Analyst',
                'Computer Vision Engineer', 'Software Engineer', 'Cloud Architect',
                'DevOps Engineer', 'Full Stack Developer', 'Cybersecurity Analyst'
            ],
            'skills': [
                'Python SQL Machine Learning Statistics Deep Learning TensorFlow',
                'Python TensorFlow PyTorch MLOps Docker Kubernetes Scikit-learn',
                'Python Django Flask REST APIs PostgreSQL Redis Git',
                'SQL Excel Power BI Tableau Python Statistics Data Visualization',
                'Research Machine Learning NLP Deep Learning Python Publications',
                'Python Java REST APIs Microservices PostgreSQL Docker Redis',
                'Python NLP BERT Transformers SpaCy Machine Learning Text Mining',
                'SQL Python Spark Hadoop ETL Airflow AWS Data Pipelines',
                'SQL Power BI Tableau Excel Business Analytics Reporting Dashboard',
                'Python OpenCV Deep Learning TensorFlow Image Processing CUDA',
                'Java Python JavaScript Algorithms Data Structures System Design',
                'AWS Azure GCP Terraform Docker Kubernetes Infrastructure Security',
                'Jenkins Docker Kubernetes Linux CI/CD Bash Terraform Monitoring',
                'React Node.js Python JavaScript SQL MongoDB REST GraphQL',
                'Network Security Penetration Testing SIEM Firewalls Python Risk'
            ],
            'jobdescription': [
                'Build predictive models and analyze large datasets to drive business decisions using statistical methods.',
                'Design and deploy scalable ML pipelines and models in production with robust monitoring.',
                'Develop web applications and APIs using Python frameworks with focus on performance.',
                'Analyze business data to extract insights using visualization tools and statistical analysis.',
                'Conduct cutting-edge AI research and publish findings in top conferences.',
                'Build scalable backend services and APIs for high-traffic applications.',
                'Develop NLP solutions for text classification, entity recognition, and language generation.',
                'Design and maintain data pipelines for ingesting and processing large-scale datasets.',
                'Create interactive dashboards and reports to support business decision-making.',
                'Develop computer vision algorithms for object detection and image classification.',
                'Solve complex engineering problems and build reliable software systems at scale.',
                'Design and implement cloud infrastructure for enterprise-level applications.',
                'Manage CI/CD pipelines and ensure reliable software delivery and infrastructure.',
                'Build end-to-end web applications with modern frontend and backend technologies.',
                'Protect organizational systems from cyber threats through security audits.'
            ],
            'advertiserurl': [
                'https://careers.google.com', 'https://jobs.microsoft.com',
                'https://amazon.jobs', 'https://careers.meta.com',
                'https://openai.com/careers', 'https://careers.apple.com',
                'https://jobs.netflix.com', 'https://careers.uber.com',
                'https://jobs.airbnb.com', 'https://careers.tesla.com',
                'https://careers.ibm.com', 'https://jobs.salesforce.com',
                'https://careers.twitter.com', 'https://jobs.spotify.com',
                'https://careers.adobe.com'
            ]
        }
        df = pd.DataFrame(demo)

    # Keep only needed columns
    cols_needed = ['jobtitle', 'skills', 'jobdescription', 'advertiserurl']
    for c in cols_needed:
        if c not in df.columns:
            df[c] = ''
    df = df[cols_needed]

    # Clean
    for c in cols_needed:
        df[c] = df[c].fillna('')

    # Combine features
    df['combine_text'] = (
        df['jobtitle'] + ' ' + df['skills'] + ' ' +
        df['jobdescription'] + ' ' + df['advertiserurl']
    )

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combine_text'])

    return df, tfidf, tfidf_matrix


def recommend_jobs(user_skills, df, tfidf, tfidf_matrix, top_n=5):
    """Return top N recommended jobs with similarity scores."""
    input_vector = tfidf.transform([user_skills])
    scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][['jobtitle', 'skills', 'jobdescription', 'advertiserurl']].copy()
    results['score'] = scores[top_indices]
    results = results.reset_index(drop=True)
    return results


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">⚙ Configuration</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"],
        help="Upload your job_us_sample.csv or any job dataset with columns: jobtitle, skills, jobdescription, advertiserurl"
    )

    st.markdown("---")

    top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    sort_by = st.selectbox(
        "Sort Results By",
        ["Similarity Score (High → Low)", "Similarity Score (Low → High)", "Job Title (A–Z)"]
    )

    st.markdown("---")
    st.markdown('<p class="sidebar-title">💡 Skill Suggestions</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-hint">Click to copy skill keywords into your query.</p>', unsafe_allow_html=True)

    suggestion_groups = {
        "Data Science": "Python Machine Learning Statistics SQL TensorFlow",
        "Web Dev": "JavaScript React Node.js REST APIs PostgreSQL",
        "DevOps": "Docker Kubernetes CI/CD Terraform AWS Linux",
        "NLP / AI": "NLP BERT Transformers Deep Learning Python Research",
        "Data Engineering": "Spark Hadoop Airflow ETL SQL Python Kafka",
    }

    for label, skills in suggestion_groups.items():
        if st.button(f"+ {label}", key=label):
            st.session_state['prefill_skills'] = skills

    st.markdown("---")
    st.markdown('<p class="sidebar-hint">Built with TF-IDF + Cosine Similarity · Streamlit</p>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
df, tfidf, tfidf_matrix = load_and_train(uploaded_file)


# ──────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────
st.markdown('<h1 class="hero-title">AI Job Recommender</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Paste your skills → get precision-matched job recommendations powered by TF-IDF & Cosine Similarity</p>',
    unsafe_allow_html=True
)
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# STATS ROW
# ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, label in zip(
    [c1, c2, c3, c4],
    [len(df), tfidf_matrix.shape[1], "TF-IDF", "Cosine"],
    ["Total Jobs", "Vocabulary Size", "Vectorizer", "Similarity"]
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num">{num}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SKILL INPUT
# ──────────────────────────────────────────────
default_skills = st.session_state.get('prefill_skills', '')

col_input, col_btn = st.columns([3, 1])
with col_input:
    st.markdown('<p class="section-heading">🔍 Enter Your Skills</p>', unsafe_allow_html=True)
    user_skills = st.text_area(
        "",
        value=default_skills,
        height=120,
        placeholder="e.g.  Python  Machine Learning  SQL  Deep Learning  TensorFlow  Statistics",
        label_visibility="collapsed"
    )

with col_btn:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    search_clicked = st.button("🎯  Find Jobs", use_container_width=True)


# ──────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

if search_clicked or user_skills.strip():
    if not user_skills.strip():
        st.warning("Please enter at least one skill to get recommendations.")
    else:
        with st.spinner("Calculating similarity across all jobs…"):
            results = recommend_jobs(user_skills, df, tfidf, tfidf_matrix, top_n=top_n)

        # Sort
        if sort_by == "Similarity Score (Low → High)":
            results = results.sort_values('score', ascending=True).reset_index(drop=True)
        elif sort_by == "Job Title (A–Z)":
            results = results.sort_values('jobtitle').reset_index(drop=True)

        # Summary
        best = results.iloc[0]['score']
        avg = results['score'].mean()

        st.markdown(f'<p class="section-heading">✨ Top {len(results)} Recommendations for  <span style="color:#00f5d4">"{user_skills[:40]}{"…" if len(user_skills)>40 else ""}"</span></p>', unsafe_allow_html=True)

        sm1, sm2 = st.columns(2)
        with sm1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-num">{best:.0%}</div>
                <div class="metric-label">Best Match Score</div>
            </div>""", unsafe_allow_html=True)
        with sm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-num">{avg:.0%}</div>
                <div class="metric-label">Avg Match Score</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        rank_classes = ['rank-1', 'rank-2', 'rank-3', 'rank-4', 'rank-5']
        rank_labels = ['🥇 Best Match', '🥈 Second Pick', '🥉 Third Choice', '4th Recommendation', '5th Recommendation']

        for i, row in results.iterrows():
            rank_cls = rank_classes[i] if i < 5 else 'rank-5'
            rank_lbl = rank_labels[i] if i < 5 else f'#{i+1} Recommendation'
            score_pct = f"{row['score']:.0%}"

            # Skills as tags
            skill_list = [s.strip() for s in str(row['skills']).split() if s.strip()][:10]
            skill_tags = ''.join([f'<span class="skill-tag">{s}</span>' for s in skill_list])

            # URL display
            url = row['advertiserurl']
            url_display = url if url else '#'
            url_label = url[:45] + '…' if len(url) > 45 else url if url else 'No link available'

            # Progress bar (score)
            score_bar_width = int(row['score'] * 100)

            st.markdown(f"""
            <div class="job-card {rank_cls}">
                <div class="rank-badge">{rank_lbl}</div>
                <div class="job-title">{row['jobtitle']}</div>
                <span class="score-badge">⚡ {score_pct} match</span>
                <div style="background:#1a1a2e;border-radius:4px;height:5px;margin-bottom:1rem;overflow:hidden;">
                    <div style="width:{score_bar_width}%;height:100%;background:linear-gradient(90deg,#00f5d4,#7b61ff);border-radius:4px;"></div>
                </div>
                <div class="skills-label">Required Skills</div>
                <div style="margin-bottom:0.9rem;">{skill_tags if skill_tags else '<span class="skill-tag">N/A</span>'}</div>
                <div class="skills-label">Description</div>
                <div style="font-size:0.85rem;color:#8888aa;line-height:1.55;margin-bottom:0.7rem;">{str(row['jobdescription'])[:220]}{'…' if len(str(row['jobdescription']))>220 else ''}</div>
                <a href="{url_display}" target="_blank" class="apply-link">→ Apply at {url_label}</a>
            </div>
            """, unsafe_allow_html=True)

        # Score distribution chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-heading">📊 Match Score Distribution</p>', unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            'Job Title': [r[:25]+'…' if len(r)>25 else r for r in results['jobtitle']],
            'Match Score (%)': (results['score'] * 100).round(1)
        })
        st.bar_chart(chart_df.set_index('Job Title'))

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🎯</div>
        <div class="empty-title">Enter your skills to discover matching jobs</div>
        <div style="font-size:0.85rem;color:#333355;margin-top:0.5rem;">
            Try: <em>Python, Machine Learning, SQL, Deep Learning</em>
        </div>
    </div>
    """, unsafe_allow_html=True)