"""
AI Career Mentor — Streamlit Dashboard
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from career_mentor_model import (
    CareerRecommendationSystem,
    generate_synthetic_dataset,
    CAREER_PROFILES,
    ALL_SKILLS,
    ALL_INTERESTS,
)

st.set_page_config(
    page_title="AI Career Mentor",
    page_icon="🎓",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 800; color: #1e3a5f; }
    .subtitle   { color: #666; font-size: 1.1rem; }
    .career-card {
        background: linear-gradient(135deg, #f0f7ff 0%, #e8f4f8 100%);
        border-left: 5px solid #2196F3;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    df = generate_synthetic_dataset(n_samples=2000)
    system = CareerRecommendationSystem(n_clusters=8)
    system.train(df)
    return system, df


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 AI Career Mentor")
    st.markdown("---")
    st.markdown("### 📋 Your Profile")

    gpa = st.slider("GPA", 2.0, 4.0, 3.2, 0.1)
    math_score = st.slider("Math Score (%)", 0, 100, 78)
    cs_score = st.slider("CS Score (%)", 0, 100, 80)
    projects = st.slider("Projects Completed", 0, 10, 2)
    certifications = st.slider("Certifications", 0, 10, 1)
    internships = st.slider("Internships", 0, 5, 0)

    st.markdown("### 🛠️ Skills")
    selected_skills = st.multiselect(
        "Select your skills:",
        options=ALL_SKILLS,
        default=["python", "sql", "statistics", "data_visualization"],
    )

    st.markdown("### 💡 Interests")
    selected_interests = st.multiselect(
        "Select your interests:",
        options=ALL_INTERESTS,
        default=["data_analysis", "research"],
    )

    st.markdown("### 📄 Resume Text (Optional)")
    resume_text = st.text_area(
        "Paste resume or project description:",
        placeholder="e.g. I built a machine learning model using Python and TensorFlow...",
        height=100,
    )

    analyze_btn = st.button("🔍 Analyze My Career Path", type="primary", use_container_width=True)

# ── Main Page ─────────────────────────────────────────────────
st.markdown('<div class="main-title">🎓 AI Career Mentor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent Career Guidance powered by Machine Learning</div>', unsafe_allow_html=True)
st.markdown("---")

with st.spinner("Loading AI Models (Random Forest · KNN · K-Means · NLP)..."):
    system, training_df = load_system()

if analyze_btn or True:  # Always show on load with defaults
    # Extract extra skills from resume
    bonus_skills = []
    if resume_text.strip():
        bonus_skills = system.extract_skills_from_text(resume_text)
        if bonus_skills:
            st.info(f"📄 **Skills extracted from resume:** {', '.join(bonus_skills)}")

    all_skills = list(set(selected_skills + bonus_skills))

    student = {
        "gpa": gpa,
        "skills": all_skills,
        "interests": selected_interests,
        "math_score": math_score,
        "cs_score": cs_score,
        "projects": projects,
        "certifications": certifications,
        "internships": internships,
    }

    result = system.recommend(student, top_n=5, training_df=training_df)
    recs = result["top_recommendations"]
    all_scores = result["all_scores"]

    # ── Row 1: Quick Metrics ──────────────────────────────────
    top = recs[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Top Career Match", top["career"])
    c2.metric("📊 Match Score", f"{top['match_score']:.1f}%")
    c3.metric("🛠️ Skill Coverage", f"{top['skill_coverage']:.1f}%")
    c4.metric("👥 Peer Group", result["cluster"]["description"])

    st.markdown("---")

    # ── Row 2: Charts ─────────────────────────────────────────
    col_chart, col_radar = st.columns([1.2, 1])

    with col_chart:
        st.markdown("#### 📊 Career Match Scores (All Careers)")
        score_df = pd.DataFrame(list(all_scores.items()), columns=["Career", "Score"])
        fig_bar = px.bar(
            score_df, x="Score", y="Career", orientation="h",
            color="Score", color_continuous_scale="Blues",
            range_x=[0, 100], text="Score",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                               yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_radar:
        st.markdown("#### 🎯 Your Skill Profile vs Top Career")
        top_career = top["career"]
        required = CAREER_PROFILES[top_career]["required_skills"][:8]
        student_skill_set = set(all_skills)
        student_vals = [1 if s in student_skill_set else 0 for s in required]
        required_vals = [1] * len(required)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=required_vals, theta=required, fill="toself",
            name="Required", fillcolor="rgba(33,150,243,0.15)", line_color="blue"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=student_vals, theta=required, fill="toself",
            name="You", fillcolor="rgba(76,175,80,0.3)", line_color="green"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
            showlegend=True, height=350,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Top Career Cards (TOP 5 RECOMMENDATIONS) ────────
    st.markdown("#### 🏆 Top 5 Career Recommendations")

    for i, rec in enumerate(recs[:5]):  # Changed from [:3] to [:5] for Top 5
        with st.expander(
            f"#{i+1}  {rec['career']}  —  Match: {rec['match_score']:.1f}%  |  Skill Coverage: {rec['skill_coverage']:.1f}%  |  💰 ${rec['avg_salary']:,}/yr  |  📈 {rec['growth_rate']} Growth",
            expanded=(i == 0)
        ):
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**✅ Skills You Already Have:**")
                if rec["mastered_skills"]:
                    for s in rec["mastered_skills"]:
                        st.markdown(f"  - ✅ `{s}`")
                else:
                    st.markdown("*None matched yet*")

                st.markdown("**❌ Skills to Acquire:**")
                if rec["missing_skills"]:
                    for s in rec["missing_skills"]:
                        st.markdown(f"  - ❌ `{s}`")
                else:
                    st.markdown("*You're fully covered!* 🎉")

            with col_right:
                st.markdown("**📍 Learning Roadmap:**")
                for step in rec["roadmap"]:
                    st.markdown(f"  {step}")

            # Skill progress bar
            st.markdown(f"**Skill Coverage Progress:**")
            st.progress(rec["skill_coverage"] / 100)

    st.markdown("---")

    # ── Row 4: Similar Profiles ───────────────────────────────
    st.markdown("#### 👥 Similar Student Profiles (KNN)")
    if result["similar_profiles"]:
        sim_df = pd.DataFrame(result["similar_profiles"])
        sim_df.columns = ["Career Match", "GPA", "Skills Count", "Similarity %"]
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

    # ── Row 5: Skill Gap Heatmap ──────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔥 Skill Gap Heatmap (Top 5 Careers)")
    top5_careers = list(all_scores.keys())[:5]
    heatmap_data = []
    for career in top5_careers:
        profile_skills = CAREER_PROFILES[career]["required_skills"]
        row = {skill: (1 if skill in set(all_skills) else -1)
               if skill in profile_skills else 0
               for skill in ALL_SKILLS[:20]}
        row["career"] = career
        heatmap_data.append(row)

    heat_df = pd.DataFrame(heatmap_data).set_index("career")
    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale=["#ff6b6b", "#f0f0f0", "#51cf66"],
        aspect="auto",
        title="Green = You have it | Red = Missing | Grey = Not required",
    )
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")
st.caption("🤖 Powered by Random Forest · KNN · K-Means · TF-IDF | Built with Scikit-learn & Streamlit")