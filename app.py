import re

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# -----------------------------------
# Skills Database
# -----------------------------------

SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "flutter",
    "firebase",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "power bi",
    "excel",
    "tableau",
    "linux",
    "rest api",
    "api",
]


# -----------------------------------
# Text Cleaning
# -----------------------------------

def clean_text(text):
    """Convert text into a simpler format for analysis."""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------------
# PDF Text Extraction
# -----------------------------------

def extract_resume_text(uploaded_file):
    """Extract text from an uploaded PDF."""

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------------
# Skill Extraction
# -----------------------------------

def extract_skills(text):
    """Find known technical skills in the text."""

    text = clean_text(text)

    found_skills = []

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        if re.search(pattern, text):
            found_skills.append(skill)

    return sorted(set(found_skills))


# -----------------------------------
# TF-IDF Similarity
# -----------------------------------

def calculate_text_similarity(resume, job_description):
    """
    Compare resume and job description using
    TF-IDF and cosine similarity.
    """

    documents = [
        clean_text(resume),
        clean_text(job_description)
    ]

    vectorizer = TfidfVectorizer(stop_words="english")

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)


# -----------------------------------
# Resume Analysis
# -----------------------------------

def analyze_resume(resume_text, job_description):

    resume_skills = extract_skills(resume_text)

    job_skills = extract_skills(job_description)

    matching_skills = sorted(
        set(resume_skills) & set(job_skills)
    )

    missing_skills = sorted(
        set(job_skills) - set(resume_skills)
    )

    if job_skills:

        skill_match = (
            len(matching_skills)
            / len(job_skills)
        ) * 100

    else:

        skill_match = 0

    similarity_score = calculate_text_similarity(
        resume_text,
        job_description
    )

    # Final score:
    # 60% technical skill match
    # 40% overall text similarity

    final_score = (
        skill_match * 0.60
        + similarity_score * 0.40
    )

    return {
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "skill_match": round(skill_match, 2),
        "similarity": similarity_score,
        "final_score": round(final_score, 2)
    }


# -----------------------------------
# User Interface
# -----------------------------------

st.title("📄 AI Resume Analyzer")

st.markdown(
    """
    ### Analyze how well your resume matches a job description.

    Upload your resume and paste a job description to discover:

    - Technical skills detected
    - Matching skills
    - Missing skills
    - Resume-to-job similarity
    - Overall match score
    """
)

st.divider()


# -----------------------------------
# Upload Resume
# -----------------------------------

st.subheader("1️⃣ Upload Your Resume")

resume_file = st.file_uploader(
    "Choose a PDF resume",
    type=["pdf"]
)


# -----------------------------------
# Job Description
# -----------------------------------

st.subheader("2️⃣ Paste Job Description")

job_description = st.text_area(
    "Job Description",
    height=250,
    placeholder="Paste the job description here..."
)


# -----------------------------------
# Analyze Button
# -----------------------------------

if st.button(
    "🔍 Analyze Resume",
    type="primary"
):

    if resume_file is None:

        st.warning(
            "Please upload your resume first."
        )

    elif not job_description.strip():

        st.warning(
            "Please paste a job description first."
        )

    else:

        with st.spinner(
            "Analyzing your resume..."
        ):

            try:

                resume_text = extract_resume_text(
                    resume_file
                )

                if not resume_text.strip():

                    st.error(
                        "Could not extract text from "
                        "this PDF. Please try another PDF."
                    )

                else:

                    results = analyze_resume(
                        resume_text,
                        job_description
                    )

                    st.success(
                        "Resume analysis completed!"
                    )

                    st.divider()

                    # --------------------------------
                    # Scores
                    # --------------------------------

                    st.subheader(
                        "📊 Analysis Results"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Overall Match",
                            f"{results['final_score']}%"
                        )

                    with col2:

                        st.metric(
                            "Skill Match",
                            f"{results['skill_match']}%"
                        )

                    with col3:

                        st.metric(
                            "Text Similarity",
                            f"{results['similarity']}%"
                        )

                    st.divider()

                    # --------------------------------
                    # Matching / Missing Skills
                    # --------------------------------

                    col1, col2 = st.columns(2)

                    with col1:

                        st.subheader(
                            "✅ Matching Skills"
                        )

                        if results[
                            "matching_skills"
                        ]:

                            for skill in results[
                                "matching_skills"
                            ]:

                                st.success(skill)

                        else:

                            st.info(
                                "No matching skills detected."
                            )

                    with col2:

                        st.subheader(
                            "⚠️ Missing Skills"
                        )

                        if results[
                            "missing_skills"
                        ]:

                            for skill in results[
                                "missing_skills"
                            ]:

                                st.warning(skill)

                        else:

                            st.success(
                                "No missing skills detected."
                            )

                    st.divider()

                    # --------------------------------
                    # Resume Skills
                    # --------------------------------

                    st.subheader(
                        "🧠 Skills Found in Resume"
                    )

                    if results["resume_skills"]:

                        st.write(
                            ", ".join(
                                results["resume_skills"]
                            )
                        )

                    else:

                        st.info(
                            "No known technical skills "
                            "were detected."
                        )

                    # --------------------------------
                    # Job Skills
                    # --------------------------------

                    st.subheader(
                        "🎯 Skills Required by Job"
                    )

                    if results["job_skills"]:

                        st.write(
                            ", ".join(
                                results["job_skills"]
                            )
                        )

                    else:

                        st.info(
                            "No known technical skills "
                            "were detected in the job description."
                        )

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )


# -----------------------------------
# Footer
# -----------------------------------

st.divider()

st.caption(
    "AI Resume Analyzer • Built with Python, "
    "Streamlit, NLP and Machine Learning"
)
