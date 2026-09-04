import json
import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from docx import Document

MODEL = "gemini-3.6-flash"

st.set_page_config(
    page_title="AI Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Resume ATS Analyzer")
st.caption("Upload a resume and get an ATS-style score, problems, and actionable improvements.")

def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")

def extract_docx_text(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        doc = Document(tmp_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def clean_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def analyze_resume(uploaded_file, job_description):
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets or set it as an environment variable."
        )

    client = genai.Client(api_key=api_key)

    instructions = f"""
You are an expert ATS resume evaluator and professional recruiter.

Evaluate the uploaded resume using an ATS-style rubric. The score is an ESTIMATE, not a score from any real ATS vendor.

Score these categories from 0 to 100:
1. keyword_match
2. formatting_and_ats_readability
3. experience_and_achievements
4. skills
5. education_and_certifications
6. clarity_and_grammar

Calculate the overall_score as a weighted average:
- keyword_match: 25%
- formatting_and_ats_readability: 20%
- experience_and_achievements: 20%
- skills: 15%
- education_and_certifications: 10%
- clarity_and_grammar: 10%

If a job description is supplied, use it to judge keyword relevance. If it is not supplied, judge against general ATS best practices for the resume's apparent target field.

Identify:
- strengths: 3 to 6 concise items
- critical_issues: the most important problems
- improvements: specific, practical changes, each with priority (High/Medium/Low)
- missing_keywords: only keywords that are reasonably supported by the target job description or the resume's apparent role; do not invent qualifications
- formatting_checks: a list of checks with name, status (Pass/Warning/Fail), and explanation
- rewritten_summary: an improved professional summary, but do not invent experience, education, metrics, or skills.

Return ONLY valid JSON with exactly this structure:
{{
  "overall_score": 0,
  "score_label": "Needs Improvement",
  "category_scores": {{
    "keyword_match": 0,
    "formatting_and_ats_readability": 0,
    "experience_and_achievements": 0,
    "skills": 0,
    "education_and_certifications": 0,
    "clarity_and_grammar": 0
  }},
  "strengths": [],
  "critical_issues": [],
  "improvements": [
    {{"priority": "High", "issue": "", "recommendation": ""}}
  ],
  "missing_keywords": [],
  "formatting_checks": [
    {{"name": "", "status": "Pass", "explanation": ""}}
  ],
  "rewritten_summary": ""
}}

Important:
- Never fabricate facts about the candidate.
- Keep the score internally consistent with the category scores and weights.
- Be strict: a polished-looking resume should not automatically receive a high ATS score.
- Do not penalize a resume merely because it lacks a photo, graphics, or personal details that are not ATS necessities.
- Do not make hiring decisions or claims about whether the candidate will get a job.

Target job description:
{job_description.strip() if job_description.strip() else "Not provided"}
"""

    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    temp_path = None
    uploaded_to_gemini = None

    try:
        if suffix == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            uploaded_to_gemini = client.files.upload(file=temp_path)
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_text(text=instructions),
                    types.Part.from_uri(
                        file_uri=uploaded_to_gemini.uri,
                        mime_type=uploaded_to_gemini.mime_type,
                    ),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    
                    max_output_tokens=5000,
                ),
            )
        elif suffix == ".docx":
            resume_text = extract_docx_text(file_bytes)
            if not resume_text.strip():
                raise ValueError("The DOCX file appears to contain no readable text.")

            response = client.models.generate_content(
                model=MODEL,
                contents=f"{instructions}\n\nRESUME TEXT:\n{resume_text}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",

                    max_output_tokens=5000,
                ),
            )
        elif suffix == ".txt":
            resume_text = file_bytes.decode("utf-8", errors="replace")
            if not resume_text.strip():
                raise ValueError("The TXT file is empty.")

            response = client.models.generate_content(
                model=MODEL,
                contents=f"{instructions}\n\nRESUME TEXT:\n{resume_text}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
        
                    max_output_tokens=5000,
                ),
            )
        else:
            raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        result = clean_json(response.text)

        # Basic validation before displaying model output.
        required = ["overall_score", "category_scores", "strengths",
                    "critical_issues", "improvements", "missing_keywords",
                    "formatting_checks", "rewritten_summary"]
        missing = [key for key in required if key not in result]
        if missing:
            raise ValueError(f"Gemini returned incomplete JSON. Missing: {', '.join(missing)}")

        result["overall_score"] = max(0, min(100, int(result["overall_score"])))
        return result

    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

st.sidebar.header("Settings")
st.sidebar.info("Gemini 2.5 Flash is used for resume analysis.")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    help="PDF is recommended. DOCX and TXT are also supported.",
)

job_description = st.text_area(
    "Target job description (optional)",
    height=180,
    placeholder="Paste the job description here for a more targeted ATS keyword analysis...",
)

if uploaded_file:
    st.write(f"**Selected:** {uploaded_file.name}")

if st.button("🔍 Analyze Resume", type="primary", disabled=uploaded_file is None):
    with st.spinner("Analyzing your resume with Gemini Flash..."):
        try:
            result = analyze_resume(uploaded_file, job_description)
            st.session_state["resume_result"] = result
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

result = st.session_state.get("resume_result")

if result:
    st.divider()
    st.subheader("ATS-Style Score")

    score = result["overall_score"]
    if score >= 85:
        label = "Excellent"
    elif score >= 70:
        label = "Good"
    elif score >= 50:
        label = "Needs Improvement"
    else:
        label = "Weak"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Overall Score", f"{score}/100")
    with c2:
        st.progress(score / 100)
        st.write(f"**{label}**")

    st.subheader("Category Scores")
    category_labels = {
        "keyword_match": "Keyword Match",
        "formatting_and_ats_readability": "ATS Readability",
        "experience_and_achievements": "Experience & Achievements",
        "skills": "Skills",
        "education_and_certifications": "Education & Certifications",
        "clarity_and_grammar": "Clarity & Grammar",
    }

    cols = st.columns(3)
    for i, (key, label_text) in enumerate(category_labels.items()):
        value = int(result["category_scores"].get(key, 0))
        with cols[i % 3]:
            st.metric(label_text, f"{value}/100")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🚨 Issues", "💡 Improvements", "🔑 Keywords", "📝 Better Summary"]
    )

    with tab1:
        if result["critical_issues"]:
            for issue in result["critical_issues"]:
                st.error(issue)
        else:
            st.success("No major issues were identified.")

        st.subheader("Strengths")
        for strength in result["strengths"]:
            st.success(strength)

        st.subheader("Formatting Checks")
        for check in result["formatting_checks"]:
            status = check.get("status", "Warning")
            message = f"**{check.get('name', 'Check')}** — {check.get('explanation', '')}"
            if status == "Pass":
                st.success(message)
            elif status == "Fail":
                st.error(message)
            else:
                st.warning(message)

    with tab2:
        for item in result["improvements"]:
            priority = item.get("priority", "Medium")
            st.markdown(
                f"**{priority} Priority — {item.get('issue', '')}**\n\n"
                f"{item.get('recommendation', '')}"
            )
            st.divider()

    with tab3:
        keywords = result["missing_keywords"]
        if keywords:
            st.write(", ".join(keywords))
        else:
            st.success("No important missing keywords were identified.")

    with tab4:
        st.write(result["rewritten_summary"])

    st.caption(
        "Important: This is an AI-based ATS-style estimate. Real applicant tracking systems "
        "vary by employer and configuration."
    )
