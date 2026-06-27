
import json
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# FILE_PATH = r"C:\Users\Abhipsa\Desktop\Gen AI\.venv\candidates.jsonl"
# OUTPUT_CSV = r"C:\Users\Abhipsa\Desktop\Gen AI\.venv\top_100_candidates.csv"

CURRENT_YEAR = datetime.now().year

JOB_DESCRIPTION = """
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years (see "what we mean by this" below)

Let's be honest about this role
We're going to write this JD differently from most. We're a Series A company that just raised our round and we're building a new AI Engineering org from scratch. This is the kind of role where the JD changes every six months because the company changes every six months. So instead of pretending we have a fixed checklist, we're going to tell you what we actually need and what we've gotten wrong before.
If you've spent your career at Google or Meta and you want a well-scoped role with a defined ladder, this isn't it.
If you've spent your career bouncing between early-stage startups and you want to "just code" without having to think about product or recruiter workflows or eval frameworks, this also isn't it.
We need someone who is simultaneously comfortable with two things that sound contradictory:
1.	Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
2.	Scrappy product-engineering attitude — willing to ship a working ranker in a week even if the underlying ML is "obviously suboptimal," because we need to learn from real users before we know what to actually optimize for.
These are not contradictory in real life. They feel contradictory because of how engineering culture sorted itself into "researcher" vs "shipper" archetypes. We need both modes available in the same person, and we'd rather you tilt slightly toward shipper than toward researcher.

What you'd actually be doing
The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles.
In practical terms, your first 90 days will probably look like:
•	Weeks 1-3: Audit what we currently have (it's mostly BM25 + rule-based scoring, working but not great). Identify the 3-4 highest-leverage things to fix.
•	Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call.
•	Weeks 9-12: Set up the evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops — so we can keep improving without flying blind.
Beyond that, you'll be driving the long-term architecture of how we do candidate-JD matching at scale, mentoring the next round of hires (we're growing the team from 4 to 12 engineers in the next year), and working closely with our recruiter-experience PM on what to build.

What we mean by "5-9 years"
This is a range, not a requirement. Some people hit "senior engineer" judgment at 4 years; some never hit it after 15. We've used 5-9 because it's roughly where people we've hired into this kind of role have landed, but we'll seriously consider candidates outside the band if other signals are strong.
That said, here are the disqualifiers we actually apply:
•	If you've spent your career in pure research environments (academic labs, research-only roles) without any production deployment — we will not move forward. We are explicit about this. We've tried it twice and it didn't work for either side.
•	If your "AI experience" consists primarily of recent (under 12 months) projects using LangChain to call OpenAI — we will probably not move forward, unless you can demonstrate substantial pre-LLM-era ML production experience. We're looking for people who understood retrieval and ranking before it became fashionable.
•	If you are a senior engineer who hasn't written production code in the last 18 months because you've moved into "architecture" or "tech lead" roles — we will probably not move forward. This role writes code.

The skills inventory (please read carefully)
Most JDs list 20 skills and you're supposed to have all of them. We're going to do this differently.
Things you absolutely need
•	Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model — we care that you've handled embedding drift, index refresh, retrieval-quality regression in production.
•	Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar. Again, the specific tech doesn't matter; the operational experience does.
•	Strong Python. Yes really, we care about code quality.
•	Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful.
Things we'd like you to have but won't reject you for
•	LLM fine-tuning experience (LoRA, QLoRA, PEFT)
•	Experience with learning-to-rank models (XGBoost-based or neural)
•	Prior exposure to HR-tech, recruiting tech, or marketplace products
•	Background in distributed systems or large-scale inference optimization
•	Open-source contributions in the AI/ML space
Things we explicitly do NOT want
This is the section most JDs skip but we think it's the most important:
•	Title-chasers. If your career trajectory shows you optimizing for "Senior" → "Staff" → "Principal" titles by switching companies every 1.5 years, we're not a fit. We need someone who plans to be here for 3+ years.
•	Framework enthusiasts. If your GitHub is full of LangChain tutorials and your blog posts are "How I used [hot framework] to build [demo]" — that's fine but it's not what we need. We need people who think about systems, not frameworks.
•	People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career. We've had bad fit experiences in both directions. If you're currently at one of these companies but have prior product-company experience, that's fine.
•	People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure. We respect your work but you'd be re-learning fundamentals here.
•	People whose work has been entirely on closed-source proprietary systems for 5+ years without external validation (papers, talks, open-source). We need to see how you think, not just trust that you can think.

On location, comp, and logistics
•	Location: Pune/Noida-preferred but flexible. We have offices in Noida and Pune(mostly used Tue/Thu). We don't require any specific number of in-office days but we expect quarterly travel for offsites. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply. Outside India: case-by-case, but we don't sponsor work visas.
•	Notice period: We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher.

The vibe check
We genuinely believe culture-fit matters more at this stage than skills-fit. Skills are teachable; the rest mostly isn't.
We work async-first and write a lot. If you find writing painful, you'll find this role painful.
We disagree openly and decide quickly. If you find that style abrasive, you'll find this role abrasive.
We move fast and break things, with the caveat that "things" are usually our internal assumptions, not user-facing systems. If you need a stable, mature codebase to be productive, you'll find this role unstable.

How to read between the lines
The "ideal candidate" we're imagining is roughly:
•	6-8 years total experience, of which 4-5 are in applied ML/AI roles at product companies (not pure services).
•	Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.
•	Has strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration (when to fine-tune vs prompt) — and can defend them with reference to systems they actually built.
•	Located in or willing to relocate to Noida or Pune.
•	Active on Redrob platform (or has clear signal of being in the job market) so we can actually talk to them.
We are aware this is a narrow profile. We're not expecting to find many matches in a 100K candidate pool. We're explicitly OK with that — we'd rather see 10 great matches than 1000 maybes.

Final note for the participants of the Redrob hackathon
If you're reading this in the context of the Intelligent Candidate Discovery & Ranking Challenge:
The "right answer" to this JD is not "find candidates whose skills section contains the most AI keywords." That's a trap we've explicitly built into the dataset.
The right answer involves reasoning about the gap between what the JD says and what the JD means. A Tier 5 candidate may not use the words "RAG" or "Pinecone" in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit. A candidate who has all the AI keywords listed as skills but whose title is "Marketing Manager" is not a fit, no matter how perfect their skill list looks.
Your ranking system should also weigh behavioral signals — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately.
Good luck.
"""

JOB_DESCRIPTION_LOWER = JOB_DESCRIPTION.lower()

WANTED_SKILLS = {
    "python", "machine learning", "nlp", "sql",
    "streamlit", "scikit-learn", "pandas", "numpy",
}
WANT_PYTHON_IN_JD = "python" in JOB_DESCRIPTION_LOWER
WANT_ML_IN_JD = "machine learning" in JOB_DESCRIPTION_LOWER


def build_candidate_text(data):
    profile = data.get("profile", {}) or {}
    parts = [
        f"Name: {profile.get('anonymized_name', '')}",
        f"Headline: {profile.get('headline', '')}",
        f"Summary: {profile.get('summary', '')}",
        f"Location: {profile.get('location', '')}, {profile.get('country', '')}",
        f"Current Role: {profile.get('current_title', '')} at {profile.get('current_company', '')}",
        f"Years of Experience: {profile.get('years_of_experience', '')}",
    ]

    for job in data.get("career_history", []) or []:
        parts.append(
            f"Worked as {job.get('title', '')} at {job.get('company', '')} "
            f"({job.get('start_date', '')} to {job.get('end_date') or 'Present'}): "
            f"{job.get('description', '')}"
        )

    for edu in data.get("education", []) or []:
        parts.append(
            f"Education: {edu.get('degree', '')} in {edu.get('field_of_study', '')} "
            f"from {edu.get('institution', '')} ({edu.get('start_year', '')}-{edu.get('end_year', '')})"
        )

    skills = data.get("skills", []) or []
    if skills:
        skill_names = ", ".join(s.get("name", "") for s in skills)
        parts.append(f"Skills: {skill_names}")

    return "\n".join(parts)


def load_candidates(file_path):
    candidates = []
    with (open(file_path, "r", encoding="utf-8") if isinstance(file_path, str) else file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            candidates.append({
                "candidate_id": data.get("candidate_id", ""),
                "text": build_candidate_text(data),
                "data": data,
            })
    return candidates


def quality_score(data):
    signals = data.get("redrob_signals", {}) or {}
    score = 0.0

    score += signals.get("profile_completeness_score", 0) / 100 * 0.20

    github = signals.get("github_activity_score", -1)
    if github != -1:
        score += github / 100 * 0.15

    score += signals.get("recruiter_response_rate", 0) * 0.10
    score += signals.get("interview_completion_rate", 0) * 0.10
    score += min(signals.get("search_appearance_30d", 0), 100) / 100 * 0.10
    score += min(signals.get("saved_by_recruiters_30d", 0), 50) / 50 * 0.10

    if signals.get("verified_email"):
        score += 0.05
    if signals.get("verified_phone"):
        score += 0.05
    if signals.get("linkedin_connected"):
        score += 0.05
    if signals.get("open_to_work_flag"):
        score += 0.10

    return min(score, 1.0)


def role_fit_score(data):
    score = 0.0
    profile = data.get("profile", {}) or {}

    title = (profile.get("current_title") or "").lower()
    headline = (profile.get("headline") or "").lower()

    if WANT_PYTHON_IN_JD and "python" in title:
        score += 0.2
    if WANT_ML_IN_JD and "machine learning" in headline:
        score += 0.2
    if (profile.get("years_of_experience") or 0) >= 3:
        score += 0.2

    skill_names = {s.get("name", "").lower() for s in (data.get("skills") or [])}
    matched = len(skill_names & WANTED_SKILLS)
    score += matched / len(WANTED_SKILLS) * 0.4

    return min(score, 1.0)


def honeypot_penalty(data):
    penalty = 0.0

    profile = data.get("profile", {}) or {}
    career = data.get("career_history", []) or []
    education = data.get("education", []) or []
    skills = data.get("skills", []) or []

    total_exp = profile.get("years_of_experience", 0) or 0
    career_months = sum(job.get("duration_months", 0) or 0 for job in career)

    if abs(career_months / 12 - total_exp) > 2:
        penalty += 0.20

    for skill in skills:
        duration = skill.get("duration_months", 0) or 0
        prof = skill.get("proficiency", "")

        if prof == "expert" and duration < 12:
            penalty += 0.15
        elif prof == "advanced" and duration < 6:
            penalty += 0.08

        if duration > career_months:
            penalty += 0.15

    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 10:
        penalty += 0.25

    if education and career:
        try:
            first_job_year = min(int(job["start_date"][:4]) for job in career if job.get("start_date"))
            latest_grad = max(edu["end_year"] for edu in education if edu.get("end_year"))
            if latest_grad > first_job_year + 2:
                penalty += 0.10
        except (ValueError, KeyError):
            pass

    return min(penalty, 0.8)


def main(job_description=None, candidates=None):
    # if candidates is None:
    #     candidates = load_candidates(FILE_PATH)

    resume_texts = [c["text"] for c in candidates]
    documents = [job_description or JOB_DESCRIPTION] + resume_texts

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,             
        max_features=50_000,  
        sublinear_tf=True,  
        dtype=np.float32,     
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    
    similarity_scores = linear_kernel(job_vector, resume_vectors).flatten()

   
    rows = []
    for candidate, cosine_score in zip(candidates, similarity_scores):
        data = candidate["data"]

        quality = quality_score(data)
        fit = role_fit_score(data)
        penalty = honeypot_penalty(data)

        final_score = (
            0.60 * cosine_score
            + 0.25 * quality
            + 0.15 * fit
            - penalty
        )

        profile = data.get("profile", {}) or {}
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "name": profile.get("anonymized_name", ""),
            # "headline": profile.get("headline", ""),
            # "current_title": profile.get("current_title", ""),
            # "years_of_experience": profile.get("years_of_experience", ""),
            # "cosine_score": round(float(cosine_score), 4),
            # "quality_score": round(float(quality), 4),
            # "role_fit_score": round(float(fit), 4),
            # "honeypot_penalty": round(float(penalty), 4),
            "final_score": round(float(final_score), 4),
        })


    top_100 = sorted(rows, key=lambda r: r["final_score"], reverse=True)[:100]
    df = pd.DataFrame(top_100)
    return df




st.title("Resume Ranker")

st.header("Job Description")
Job_Description = st.text_area("Enter the job description")

st.header("Upload Resumes")
uploaded_files =  st.file_uploader("Upload files",type =["pdf","jsonl","json","txt"],accept_multiple_files=True)

if uploaded_files and Job_Description:

    candidates = [ ]

    for file in uploaded_files:
        candidates.extend(load_candidates(file))
        
    if not candidates:
        st.warning("No candidates could be parsed from the uploaded files.")
    else:
        st.header("Ranking Resumes")
        with st.spinner(f"Scoring {len(candidates)} candidates..."):
            results = main(Job_Description, candidates)

        st.dataframe(results, use_container_width=True, height=480)

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="ranked_candidates.csv",
            mime="text/csv",)
  