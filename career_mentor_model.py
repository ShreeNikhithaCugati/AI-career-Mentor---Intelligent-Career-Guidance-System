"""
AI Career Mentor – Intelligent Career Guidance System
=====================================================
ML Model Implementation
Algorithms: Random Forest, KNN, K-Means, TF-IDF/NLP
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
import json
import pickle
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────

CAREER_PROFILES = {
    "Data Scientist": {
        "required_skills": ["python", "machine_learning", "statistics", "sql", "data_visualization",
                            "deep_learning", "pandas", "numpy", "tensorflow", "r"],
        "min_gpa": 3.2,
        "preferred_interests": ["data_analysis", "research", "mathematics", "ai"],
        "avg_salary": 110000,
        "growth_rate": "Very High",
    },
    "Software Engineer": {
        "required_skills": ["python", "java", "data_structures", "algorithms", "git",
                            "system_design", "sql", "javascript", "rest_api", "docker"],
        "min_gpa": 3.0,
        "preferred_interests": ["coding", "problem_solving", "software", "technology"],
        "avg_salary": 105000,
        "growth_rate": "High",
    },
    "ML Engineer": {
        "required_skills": ["python", "machine_learning", "deep_learning", "tensorflow",
                            "pytorch", "mlops", "docker", "cloud", "statistics", "git"],
        "min_gpa": 3.3,
        "preferred_interests": ["ai", "research", "coding", "mathematics"],
        "avg_salary": 120000,
        "growth_rate": "Very High",
    },
    "Cybersecurity Analyst": {
        "required_skills": ["networking", "linux", "ethical_hacking", "cryptography",
                            "firewalls", "python", "incident_response", "vulnerability_assessment"],
        "min_gpa": 3.0,
        "preferred_interests": ["security", "networking", "hacking", "problem_solving"],
        "avg_salary": 95000,
        "growth_rate": "High",
    },
    "Web Developer": {
        "required_skills": ["html", "css", "javascript", "react", "nodejs",
                            "sql", "rest_api", "git", "typescript", "responsive_design"],
        "min_gpa": 2.8,
        "preferred_interests": ["design", "coding", "creativity", "technology"],
        "avg_salary": 85000,
        "growth_rate": "High",
    },
    "Cloud Architect": {
        "required_skills": ["aws", "azure", "docker", "kubernetes", "devops",
                            "networking", "python", "terraform", "linux", "security"],
        "min_gpa": 3.1,
        "preferred_interests": ["infrastructure", "technology", "problem_solving", "architecture"],
        "avg_salary": 130000,
        "growth_rate": "Very High",
    },
    "Data Analyst": {
        "required_skills": ["sql", "excel", "python", "data_visualization", "statistics",
                            "power_bi", "tableau", "pandas", "r", "business_intelligence"],
        "min_gpa": 2.9,
        "preferred_interests": ["data_analysis", "business", "mathematics", "reporting"],
        "avg_salary": 75000,
        "growth_rate": "High",
    },
    "Product Manager": {
        "required_skills": ["product_strategy", "agile", "user_research", "data_analysis",
                            "communication", "stakeholder_management", "roadmapping", "sql"],
        "min_gpa": 3.0,
        "preferred_interests": ["business", "leadership", "design", "technology"],
        "avg_salary": 115000,
        "growth_rate": "High",
    },
    "DevOps Engineer": {
        "required_skills": ["docker", "kubernetes", "ci_cd", "linux", "python",
                            "terraform", "aws", "monitoring", "git", "bash"],
        "min_gpa": 3.0,
        "preferred_interests": ["infrastructure", "automation", "coding", "technology"],
        "avg_salary": 100000,
        "growth_rate": "High",
    },
    "AI Research Scientist": {
        "required_skills": ["python", "deep_learning", "pytorch", "mathematics",
                            "statistics", "research_writing", "tensorflow", "nlp", "computer_vision"],
        "min_gpa": 3.7,
        "preferred_interests": ["research", "ai", "mathematics", "academia"],
        "avg_salary": 140000,
        "growth_rate": "Very High",
    },
}

ALL_SKILLS = sorted(set(
    skill for profile in CAREER_PROFILES.values()
    for skill in profile["required_skills"]
))

ALL_INTERESTS = sorted(set(
    interest for profile in CAREER_PROFILES.values()
    for interest in profile["preferred_interests"]
))

CAREER_ROADMAPS = {
    "Data Scientist": [
        "1. Master Python (NumPy, Pandas, Matplotlib)",
        "2. Learn Statistics & Probability",
        "3. Study Machine Learning (Scikit-learn)",
        "4. Deep Learning (TensorFlow / PyTorch)",
        "5. SQL & Database Skills",
        "6. Build 3-5 real projects (Kaggle competitions)",
        "7. Learn Data Visualization (Tableau / Power BI)",
        "8. Get Certifications: Google Data Analytics, IBM Data Science",
    ],
    "Software Engineer": [
        "1. Learn Data Structures & Algorithms",
        "2. Master Python / Java / C++",
        "3. Study System Design",
        "4. Learn Git & Version Control",
        "5. Build REST APIs",
        "6. Practice LeetCode (200+ problems)",
        "7. Contribute to Open Source",
        "8. Get Certifications: AWS Developer, Oracle Java",
    ],
    "ML Engineer": [
        "1. Strong Python Programming",
        "2. Machine Learning Fundamentals",
        "3. Deep Learning (CNNs, RNNs, Transformers)",
        "4. MLOps & Model Deployment",
        "5. Docker & Kubernetes",
        "6. Cloud Platforms (AWS/GCP/Azure)",
        "7. Build End-to-End ML Pipelines",
        "8. Get Certifications: AWS ML Specialty, GCP ML Engineer",
    ],
    "Cybersecurity Analyst": [
        "1. Learn Networking Fundamentals (CompTIA Network+)",
        "2. Master Linux",
        "3. Study Ethical Hacking (Kali Linux)",
        "4. Learn Cryptography Basics",
        "5. Practice on TryHackMe / HackTheBox",
        "6. Study Incident Response",
        "7. Get Certifications: CEH, CompTIA Security+, OSCP",
    ],
    "Web Developer": [
        "1. Master HTML5, CSS3, JavaScript",
        "2. Learn React / Vue / Angular",
        "3. Backend: Node.js or Django/Flask",
        "4. SQL & NoSQL Databases",
        "5. REST API Design",
        "6. Learn TypeScript",
        "7. Build 5+ Portfolio Projects",
        "8. Get Certifications: Meta Front-End Dev, freeCodeCamp",
    ],
    "Cloud Architect": [
        "1. Learn Networking Fundamentals",
        "2. Master Linux",
        "3. Study AWS / Azure / GCP",
        "4. Learn Docker & Kubernetes",
        "5. Infrastructure as Code (Terraform)",
        "6. Study Security & Compliance",
        "7. Practice Cloud Projects",
        "8. Get Certifications: AWS Solutions Architect, Azure Administrator",
    ],
    "Data Analyst": [
        "1. Master Excel & Google Sheets",
        "2. Learn SQL (Joins, Aggregations, CTEs)",
        "3. Python (Pandas, Matplotlib)",
        "4. Statistics & Data Interpretation",
        "5. Power BI or Tableau",
        "6. Business Intelligence Concepts",
        "7. Build Dashboards & Reports Portfolio",
        "8. Get Certifications: Google Data Analytics, Microsoft Power BI",
    ],
    "Product Manager": [
        "1. Learn Agile & Scrum Methodologies",
        "2. Study User Research & UX Design",
        "3. Learn SQL for Data-Driven Decisions",
        "4. Master Roadmapping Tools (Jira, Confluence)",
        "5. Study Business Strategy",
        "6. Practice Stakeholder Communication",
        "7. Work on Side Product Projects",
        "8. Get Certifications: CSPO, PMI-ACP, Google PM Certificate",
    ],
    "DevOps Engineer": [
        "1. Master Linux & Bash Scripting",
        "2. Learn Git & CI/CD (Jenkins, GitHub Actions)",
        "3. Docker & Container Orchestration",
        "4. Kubernetes",
        "5. Cloud Platforms (AWS/GCP/Azure)",
        "6. Infrastructure as Code (Terraform, Ansible)",
        "7. Monitoring & Logging (Prometheus, Grafana)",
        "8. Get Certifications: CKA, AWS DevOps Engineer",
    ],
    "AI Research Scientist": [
        "1. Master Mathematics (Linear Algebra, Calculus, Stats)",
        "2. Advanced Python & Scientific Computing",
        "3. Deep Learning Theory & Implementation",
        "4. Research Paper Reading & Reproduction",
        "5. Specialize: NLP / CV / RL",
        "6. Publish Research (arXiv, conferences)",
        "7. Contribute to Open-Source ML Libraries",
        "8. Pursue MS/PhD in AI/ML/CS",
    ],
}


def generate_synthetic_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic student career data."""
    np.random.seed(random_state)
    records = []

    careers = list(CAREER_PROFILES.keys())

    for _ in range(n_samples):
        career = np.random.choice(careers)
        profile = CAREER_PROFILES[career]

        # GPA: Gaussian around career minimum, capped [2.0, 4.0]
        gpa = np.clip(np.random.normal(profile["min_gpa"] + 0.2, 0.3), 2.0, 4.0)

        # Skills: 60-90% of required + some random noise skills
        n_required = int(np.random.uniform(0.6, 0.95) * len(profile["required_skills"]))
        student_skills = np.random.choice(profile["required_skills"], n_required, replace=False).tolist()
        n_noise = np.random.randint(0, 4)
        noise_pool = [s for s in ALL_SKILLS if s not in student_skills]
        if noise_pool:
            student_skills += np.random.choice(noise_pool, min(n_noise, len(noise_pool)), replace=False).tolist()

        # Interests: 1-2 preferred + some random
        n_pref = np.random.randint(1, len(profile["preferred_interests"]) + 1)
        interests = np.random.choice(profile["preferred_interests"], min(n_pref, len(profile["preferred_interests"])), replace=False).tolist()
        n_extra = np.random.randint(0, 2)
        interest_pool = [i for i in ALL_INTERESTS if i not in interests]
        if interest_pool:
            interests += np.random.choice(interest_pool, min(n_extra, len(interest_pool)), replace=False).tolist()

        # Academic details
        math_score = int(np.random.normal(75, 15))
        cs_score = int(np.random.normal(75, 15))
        projects = np.random.randint(0, 6)
        certifications = np.random.randint(0, 4)
        internships = np.random.randint(0, 3)

        records.append({
            "career": career,
            "gpa": round(gpa, 2),
            "skills": student_skills,
            "interests": interests,
            "math_score": np.clip(math_score, 0, 100),
            "cs_score": np.clip(cs_score, 0, 100),
            "projects": projects,
            "certifications": certifications,
            "internships": internships,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

class FeatureEngineer:
    """Converts raw student profile into ML-ready features."""

    def __init__(self):
        self.skill_binarizer = MultiLabelBinarizer(classes=ALL_SKILLS)
        self.interest_binarizer = MultiLabelBinarizer(classes=ALL_INTERESTS)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        skill_features = self.skill_binarizer.fit_transform(df["skills"])
        interest_features = self.interest_binarizer.fit_transform(df["interests"])
        numerical = df[["gpa", "math_score", "cs_score", "projects", "certifications", "internships"]].values
        numerical_scaled = self.scaler.fit_transform(numerical)
        self._fitted = True
        return np.hstack([skill_features, interest_features, numerical_scaled])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform.")
        skill_features = self.skill_binarizer.transform(df["skills"])
        interest_features = self.interest_binarizer.transform(df["interests"])
        numerical = df[["gpa", "math_score", "cs_score", "projects", "certifications", "internships"]].values
        numerical_scaled = self.scaler.transform(numerical)
        return np.hstack([skill_features, interest_features, numerical_scaled])

    def transform_single(self, student: dict) -> np.ndarray:
        """Transform a single student dict to feature vector."""
        df = pd.DataFrame([{
            "skills": student.get("skills", []),
            "interests": student.get("interests", []),
            "gpa": student.get("gpa", 3.0),
            "math_score": student.get("math_score", 70),
            "cs_score": student.get("cs_score", 70),
            "projects": student.get("projects", 0),
            "certifications": student.get("certifications", 0),
            "internships": student.get("internships", 0),
        }])
        return self.transform(df)


# ─────────────────────────────────────────────────────────────
# 3. ML MODELS
# ─────────────────────────────────────────────────────────────

class CareerRecommendationSystem:
    """
    Multi-model AI Career Recommendation System.
    - Random Forest : Primary career classification
    - KNN           : Similarity-based career matching
    - K-Means       : Student profile clustering
    - TF-IDF/NLP    : Resume / free-text skill extraction
    """

    def __init__(self, n_clusters: int = 8):
        self.feature_engineer = FeatureEngineer()
        self.label_encoder = LabelEncoder()

        # Model 1: Random Forest (primary classifier)
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Model 2: KNN (similarity-based)
        self.knn_model = KNeighborsClassifier(
            n_neighbors=7,
            metric="euclidean",
            weights="distance",
        )

        # Model 3: K-Means (student clustering)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        # Model 4: TF-IDF for resume/free-text skill extraction
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=500,
            stop_words="english",
        )

        self.careers = list(CAREER_PROFILES.keys())
        self._trained = False

    # ── Training ──────────────────────────────────────────────

    def train(self, df: pd.DataFrame):
        print("=" * 60)
        print("  AI CAREER MENTOR — MODEL TRAINING")
        print("=" * 60)
        print(f"\n📊 Dataset: {len(df)} student profiles | {df['career'].nunique()} careers\n")

        # Feature engineering
        X = self.feature_engineer.fit_transform(df)
        y = self.label_encoder.fit_transform(df["career"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── Random Forest ──
        print("🌲 Training Random Forest Classifier...")
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_cv = cross_val_score(self.rf_model, X, y, cv=5, scoring="accuracy")
        print(f"   ✅ Test Accuracy     : {rf_acc:.4f} ({rf_acc*100:.1f}%)")
        print(f"   ✅ CV Accuracy (5-fold): {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

        # ── KNN ──
        print("\n🔍 Training KNN Classifier...")
        self.knn_model.fit(X_train, y_train)
        knn_pred = self.knn_model.predict(X_test)
        knn_acc = accuracy_score(y_test, knn_pred)
        print(f"   ✅ Test Accuracy: {knn_acc:.4f} ({knn_acc*100:.1f}%)")

        # ── K-Means Clustering ──
        print("\n🗂️  Training K-Means Clustering...")
        self.kmeans.fit(X)
        print(f"   ✅ {self.kmeans.n_clusters} student clusters identified")

        # ── Classification Report ──
        print("\n📋 Random Forest Classification Report:")
        career_names = self.label_encoder.inverse_transform(sorted(set(y_test)))
        print(classification_report(y_test, rf_pred,
                                    target_names=self.label_encoder.classes_,
                                    zero_division=0))

        self._trained = True
        print("=" * 60)
        print("  ✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 60)
        return self

    # ── Career Match Score ────────────────────────────────────

    def compute_career_match_scores(self, student: dict) -> dict:
        """
        Compute match % for each career using:
        - RF probability (60%)
        - Skill overlap (30%)
        - Interest overlap (10%)
        """
        X = self.feature_engineer.transform_single(student)
        rf_probs = dict(zip(
            self.label_encoder.classes_,
            self.rf_model.predict_proba(X)[0]
        ))

        student_skills = set(student.get("skills", []))
        student_interests = set(student.get("interests", []))
        student_gpa = student.get("gpa", 0)

        scores = {}
        for career, profile in CAREER_PROFILES.items():
            required = set(profile["required_skills"])
            preferred_interests = set(profile["preferred_interests"])

            skill_overlap = len(student_skills & required) / len(required) if required else 0
            interest_overlap = len(student_interests & preferred_interests) / len(preferred_interests) if preferred_interests else 0
            gpa_score = min(1.0, student_gpa / profile["min_gpa"]) if profile["min_gpa"] > 0 else 1.0

            rf_score = rf_probs.get(career, 0.0)

            # Weighted combination
            composite = (0.55 * rf_score) + (0.30 * skill_overlap) + (0.10 * interest_overlap) + (0.05 * gpa_score)
            scores[career] = round(composite * 100, 1)

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # ── Skill Gap Analysis ────────────────────────────────────

    def skill_gap_analysis(self, student_skills: list, career: str) -> dict:
        """Returns missing skills and mastered skills for a given career."""
        required = set(CAREER_PROFILES[career]["required_skills"])
        student = set(student_skills)
        missing = sorted(required - student)
        mastered = sorted(required & student)
        extra = sorted(student - required)
        return {
            "career": career,
            "required_skills": sorted(required),
            "mastered_skills": mastered,
            "missing_skills": missing,
            "extra_skills": extra,
            "skill_coverage_pct": round(len(mastered) / len(required) * 100, 1) if required else 0,
        }

    # ── KNN Similar Profiles ──────────────────────────────────

    def find_similar_profiles(self, student: dict, top_n: int = 5, training_df: pd.DataFrame = None) -> list:
        """Find top-N most similar student profiles using KNN."""
        if training_df is None:
            return []
        X_student = self.feature_engineer.transform_single(student)
        X_all = self.feature_engineer.transform(training_df)
        distances = np.linalg.norm(X_all - X_student, axis=1)
        indices = np.argsort(distances)[:top_n]
        similar = []
        for idx in indices:
            row = training_df.iloc[idx]
            similar.append({
                "career": row["career"],
                "gpa": row["gpa"],
                "skills_count": len(row["skills"]),
                "similarity_score": round((1 / (1 + distances[idx])) * 100, 1),
            })
        return similar

    # ── Student Cluster ───────────────────────────────────────

    def get_student_cluster(self, student: dict) -> dict:
        """Assigns student to a K-Means cluster."""
        X = self.feature_engineer.transform_single(student)
        cluster_id = int(self.kmeans.predict(X)[0])
        return {"cluster_id": cluster_id, "description": f"Peer Group #{cluster_id + 1}"}

    # ── NLP Resume Skill Extraction ───────────────────────────

    def extract_skills_from_text(self, text: str) -> list:
        """Extract known skills from free-text resume or description using keyword matching + TF-IDF."""
        text_lower = text.lower().replace("-", "_").replace(" ", "_")
        found_skills = [skill for skill in ALL_SKILLS if skill in text_lower]

        # Also try partial matches
        words = set(text.lower().split())
        for skill in ALL_SKILLS:
            parts = skill.split("_")
            if all(p in words for p in parts) and skill not in found_skills:
                found_skills.append(skill)

        return sorted(set(found_skills))

    # ── Full Recommendation Pipeline ─────────────────────────

    def recommend(self, student: dict, top_n: int = 3, training_df: pd.DataFrame = None) -> dict:
        """
        Full recommendation pipeline for one student.
        Returns top careers, match scores, skill gaps, and roadmap.
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        match_scores = self.compute_career_match_scores(student)
        top_careers = list(match_scores.items())[:top_n]

        results = []
        for career, score in top_careers:
            gap = self.skill_gap_analysis(student.get("skills", []), career)
            roadmap = CAREER_ROADMAPS.get(career, [])
            results.append({
                "career": career,
                "match_score": score,
                "skill_coverage": gap["skill_coverage_pct"],
                "missing_skills": gap["missing_skills"],
                "mastered_skills": gap["mastered_skills"],
                "roadmap": roadmap,
                "avg_salary": CAREER_PROFILES[career]["avg_salary"],
                "growth_rate": CAREER_PROFILES[career]["growth_rate"],
            })

        cluster_info = self.get_student_cluster(student)
        similar = self.find_similar_profiles(student, top_n=3, training_df=training_df)

        return {
            "student_profile": student,
            "top_recommendations": results,
            "all_scores": match_scores,
            "cluster": cluster_info,
            "similar_profiles": similar,
        }

    # ── Save / Load ───────────────────────────────────────────

    def save(self, path: str = "career_mentor_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"💾 Model saved to: {path}")

    @staticmethod
    def load(path: str = "career_mentor_model.pkl") -> "CareerRecommendationSystem":
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"📂 Model loaded from: {path}")
        return model


# ─────────────────────────────────────────────────────────────
# 4. PRETTY REPORT PRINTER
# ─────────────────────────────────────────────────────────────

def print_recommendation_report(result: dict):
    """Prints a nicely formatted career recommendation report."""
    student = result["student_profile"]
    recs = result["top_recommendations"]

    print("\n" + "=" * 65)
    print("        🎓  AI CAREER MENTOR — CAREER RECOMMENDATION REPORT")
    print("=" * 65)
    print(f"\n👤 Student Profile:")
    print(f"   GPA            : {student.get('gpa', 'N/A')}")
    print(f"   Math Score     : {student.get('math_score', 'N/A')}")
    print(f"   CS Score       : {student.get('cs_score', 'N/A')}")
    print(f"   Projects       : {student.get('projects', 0)}")
    print(f"   Certifications : {student.get('certifications', 0)}")
    print(f"   Internships    : {student.get('internships', 0)}")
    print(f"   Skills         : {', '.join(student.get('skills', []))}")
    print(f"   Interests      : {', '.join(student.get('interests', []))}")
    print(f"   Peer Group     : {result['cluster']['description']}")

    print(f"\n{'─'*65}")
    print(f"🏆  TOP {len(recs)} CAREER RECOMMENDATIONS")
    print(f"{'─'*65}")

    for i, rec in enumerate(recs, 1):
        bar_len = int(rec['match_score'] / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"\n  #{i}  {rec['career']}")
        print(f"       Match Score    : {rec['match_score']:.1f}% [{bar}]")
        print(f"       Skill Coverage : {rec['skill_coverage']:.1f}%")
        print(f"       Avg Salary     : ${rec['avg_salary']:,} / year")
        print(f"       Growth Rate    : {rec['growth_rate']}")

        if rec["mastered_skills"]:
            print(f"       ✅ Mastered    : {', '.join(rec['mastered_skills'][:5])}"
                  + ("..." if len(rec['mastered_skills']) > 5 else ""))
        if rec["missing_skills"]:
            print(f"       ❌ Missing     : {', '.join(rec['missing_skills'][:5])}"
                  + ("..." if len(rec['missing_skills']) > 5 else ""))

        print(f"\n       📍 Learning Roadmap:")
        for step in rec["roadmap"][:4]:
            print(f"          {step}")
        if len(rec["roadmap"]) > 4:
            print(f"          ... +{len(rec['roadmap'])-4} more steps")

    print(f"\n{'─'*65}")
    print("📊  ALL CAREER MATCH SCORES")
    print(f"{'─'*65}")
    for career, score in result["all_scores"].items():
        bar = "█" * int(score / 4) + "░" * (25 - int(score / 4))
        print(f"  {career:<25} {score:5.1f}% [{bar}]")

    if result["similar_profiles"]:
        print(f"\n{'─'*65}")
        print("👥  SIMILAR STUDENT PROFILES (KNN)")
        print(f"{'─'*65}")
        for p in result["similar_profiles"]:
            print(f"  • Career: {p['career']:<25} GPA: {p['gpa']:.1f}  Similarity: {p['similarity_score']:.1f}%")

    print("\n" + "=" * 65)
    print("  ✅  Report Complete | AI Career Mentor System")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────
# 5. RESUME ANALYSIS DEMO
# ─────────────────────────────────────────────────────────────

def demo_resume_analysis(system: CareerRecommendationSystem):
    sample_resume = """
    I am a computer science student with strong Python programming skills.
    I have worked on machine learning projects using TensorFlow and PyTorch.
    I have experience with SQL databases, pandas, numpy, and data visualization.
    I have completed a deep learning certification and have internship experience
    in data analysis. I enjoy statistics, research, and AI development.
    """
    print("\n" + "─" * 65)
    print("📄  RESUME / FREE-TEXT SKILL EXTRACTION (NLP)")
    print("─" * 65)
    print("Input text:\n", sample_resume.strip())
    extracted = system.extract_skills_from_text(sample_resume)
    print(f"\n✅ Extracted Skills ({len(extracted)}): {', '.join(extracted)}")
    print("─" * 65 + "\n")
    return extracted


# ─────────────────────────────────────────────────────────────
# 6. MAIN DEMO
# ─────────────────────────────────────────────────────────────

def main():
    # Generate dataset
    print("🔄 Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_samples=2000)
    print(f"✅ Dataset generated: {df.shape}\n")

    # Initialize & train
    system = CareerRecommendationSystem(n_clusters=8)
    system.train(df)

    # Demo: Resume analysis to extract skills
    extracted_skills = demo_resume_analysis(system)

    # Example student profiles to test
    students = [
        {
            "name": "Alice (Aspiring Data Scientist)",
            "gpa": 3.7,
            "skills": ["python", "pandas", "statistics", "data_visualization", "numpy"],
            "interests": ["data_analysis", "research", "ai"],
            "math_score": 88,
            "cs_score": 85,
            "projects": 3,
            "certifications": 2,
            "internships": 1,
        },
        {
            "name": "Bob (Software Enthusiast)",
            "gpa": 3.2,
            "skills": ["python", "java", "git", "data_structures", "algorithms", "rest_api"],
            "interests": ["coding", "problem_solving", "technology"],
            "math_score": 80,
            "cs_score": 90,
            "projects": 4,
            "certifications": 1,
            "internships": 2,
        },
        {
            "name": "Carol (Security Focused)",
            "gpa": 3.1,
            "skills": ["networking", "linux", "python", "ethical_hacking"],
            "interests": ["security", "networking", "hacking"],
            "math_score": 72,
            "cs_score": 78,
            "projects": 2,
            "certifications": 1,
            "internships": 0,
        },
    ]

    for student in students:
        name = student.pop("name")
        print(f"\n{'='*65}")
        print(f"  🎯 Analyzing: {name}")
        print(f"{'='*65}")
        result = system.recommend(student, top_n=3, training_df=df)
        print_recommendation_report(result)
        student["name"] = name  # restore

    # Save model
    system.save("career_mentor_model.pkl")

    return system, df


if __name__ == "__main__":
    system, df = main()
