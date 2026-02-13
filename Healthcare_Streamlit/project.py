# ================= IMPORT LIBRARIES =================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Health Insurance Dashboard",
    layout="wide"
)
st.markdown("""
<style>
/* MAIN PAGE SPACING FIX */
.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 0.5rem !important;
}

/* Allow vertical scroll if needed */
html, body {
    overflow-y: auto;
}

/* Sidebar spacing */
section[data-testid="stSidebar"] > div {
    padding-top: 1rem !important;
}

/* Metric card spacing */
.metric-card {
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 16px 10px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
}

.metric-icon {
    font-size: 26px;
}

.metric-label {
    font-size: 14px;
    color: #6b7280;
    margin-top: 6px;
}

.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #111827;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ================= TITLE =================
st.markdown("## 🏥 Health Care Insurance Cost Analysis")

# ================= LOAD DATA =================
df = pd.read_csv(r"E:\IRL\IRL_Projects\ML\HealthCare\Dataset\insurance-1.csv")

# Rename columns
df.rename(columns={
    "age": "Age",
    "sex": "Gender",
    "bmi": "BMI",
    "children": "Childrens",
    "smoker": "Smoker",
    "region": "Region",
    "charges": "Charges"
}, inplace=True)

# ================= METRICS =================
st.markdown("### 🎯 Key Metrics")

c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(icon, label, value):
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

with c1:
    st.markdown(
        metric_card("🎂", "Avg Age", f"{int(df['Age'].mean())} yrs"),
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        metric_card("⚖️", "Avg BMI", f"{df['BMI'].mean():.1f}"),
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        metric_card("🚬", "Smokers", df[df["Smoker"] == "yes"].shape[0]),
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        metric_card("🚭", "Non-Smokers", df[df["Smoker"] == "no"].shape[0]),
        unsafe_allow_html=True
    )

with c5:
    st.markdown(
        metric_card("💰", "Avg Charges", f"{df['Charges'].mean():,.0f}"),
        unsafe_allow_html=True
    )

# ================= HELPER FUNCTION =================
def label_bars(ax):
    for bar in ax.patches:
        ax.annotate(
            f"{int(bar.get_height())}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10
        )

# ================= SIDEBAR =================
st.sidebar.header("EDA Controls")

analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Univariate Analysis", "Bivariate Analysis"]
)

st.subheader("📊 Visual Analysis")

# =====================================================
# ================= UNIVARIATE ========================
# =====================================================
if analysis_type == "Univariate Analysis":

    var_type = st.sidebar.selectbox(
        "Select Variable Type",
        ["Numerical", "Categorical"]
    )

    # -------- NUMERICAL --------
    if var_type == "Numerical":
        col = st.sidebar.selectbox(
            "Select Column",
            ["Age", "BMI", "Charges", "Childrens"]
        )

        fig, ax = plt.subplots(figsize=(8,3))

        if col == "Age":
            sns.histplot(df[col], bins=15, color="#6366f1", ax=ax)
            ax.set_title("Age Distribution")
            st.pyplot(fig)
            st.write("👉 Shows the age spread of policyholders.")

        elif col == "BMI":
            sns.histplot(df[col], bins=20, kde=True, color="#22c55e", ax=ax)
            ax.set_title("BMI Distribution")
            st.pyplot(fig)
            st.write("👉 Highlights health risk levels based on BMI.")

        elif col == "Charges":
            sns.histplot(df[col], bins=30, kde=True, color="#f59e0b", ax=ax)
            ax.set_title("Charges Distribution")
            st.pyplot(fig)
            st.write("👉 Displays cost imbalance among customers.")

        elif col == "Childrens":
            sns.countplot(x=df[col], color="#0ea5e9", ax=ax)
            label_bars(ax)
            ax.set_title("Children Count")
            st.pyplot(fig)
            st.write("👉 Most customers have fewer dependents.")

    # -------- CATEGORICAL --------
    else:
        col = st.sidebar.selectbox(
            "Select Column",
            ["Gender", "Smoker", "Region"]
        )

        if col == "Gender":
            fig, ax = plt.subplots(figsize=(8,3))
            sns.countplot(x=df[col], color="#6366f1", ax=ax)
            label_bars(ax)
            ax.set_title("Gender Distribution")
            st.pyplot(fig)
            st.write("👉 Gender distribution is balanced.")

        elif col == "Smoker":
            fig, ax = plt.subplots(figsize=(8,3))
            sns.countplot(x=df[col], palette=["#22c55e", "#ef4444"], ax=ax)
            label_bars(ax)
            ax.set_title("Smoking Status")
            st.pyplot(fig)
            st.write("👉 Smokers form a high-risk group.")

        elif col == "Region":
            fig, ax = plt.subplots(figsize=(8,3))
            sns.countplot(x=df[col], palette="Set2", ax=ax)
            label_bars(ax)
            ax.set_title("Region Distribution")
            st.pyplot(fig)
            st.write("👉 Customers are distributed across regions.")

# =====================================================
# ================= BIVARIATE =========================
# =====================================================
else:

    bi_type = st.sidebar.selectbox(
        "Select Relationship",
        ["Num vs Num", "Cat vs Cat", "Num vs Cat"]
    )

    # -------- NUM vs NUM --------
    if bi_type == "Num vs Num":
        plot = st.sidebar.selectbox(
            "Select Plot",
            ["Charges vs Age", "Charges vs BMI"]
        )

        fig, ax = plt.subplots(figsize=(8,3))

        if plot == "Charges vs Age":
            sns.lineplot(
                x=df["Age"],
                y=df["Charges"],
                estimator="mean",
                errorbar=None,
                color="#22c55e",
                linewidth=2.5,
                ax=ax
            )
            ax.set_title("Average Charges by Age")
            st.pyplot(fig)
            st.write("👉 Charges increase steadily with age.")

        elif plot == "Charges vs BMI":
            sns.scatterplot(
                x=df["BMI"],
                y=df["Charges"],
                alpha=0.5,
                color="#0ea5e9",
                ax=ax
            )
            sns.regplot(
                x=df["BMI"],
                y=df["Charges"],
                scatter=False,
                color="#ef4444",
                ax=ax
            )
            ax.set_title("Charges vs BMI")
            st.pyplot(fig)
            st.write("👉 Higher BMI is linked to higher charges.")

    # -------- CAT vs CAT --------
    elif bi_type == "Cat vs Cat":
        fig, ax = plt.subplots(figsize=(8,3))
        sns.countplot(
            x=df["Smoker"],
            hue=df["Gender"],
            palette="Set2",
            ax=ax
        )
        ax.set_title("Smoker vs Gender")
        st.pyplot(fig)
        st.write("👉 Smoking patterns vary slightly by gender.")

    # -------- NUM vs CAT --------
    else:
        fig, ax = plt.subplots(figsize=(8,3))
        sns.barplot(
            x=df["Smoker"],
            y=df["Charges"],
            palette=["#22c55e", "#ef4444"],
            ax=ax
        )
        ax.set_title("Charges vs Smoking Status")
        st.pyplot(fig)
        st.write("👉 Smokers pay significantly higher charges.")

# steps to run
# 1.cd C:\Users\Venkstesh\VS Projects\Healthcare_Streamlit>
# 2. "C:/Users/Venkstesh/VS Projects/.venv/Scripts/Activate.ps1"
# 3. dir
# 4. streamlit run project.py