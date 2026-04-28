import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Customer Behavior Dashboard", page_icon="📊", layout="wide")

st.title("📊 Customer Purchasing Behavior Dashboard")
st.write("Interactive analysis of customer purchasing patterns, loyalty, and income trends.")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Customer-Purchasing-Behaviors.csv")
    return df

df = load_data()

# ---------------------------
# BASIC INFO
# ---------------------------
st.subheader("Dataset Overview")
st.dataframe(df.head())

st.write("Shape:", df.shape)

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------
st.sidebar.header("Filters")

regions = st.sidebar.multiselect(
    "Select Region",
    options=df["region"].unique(),
    default=df["region"].unique()
)

income_range = st.sidebar.slider(
    "Income Range",
    int(df["annual_income"].min()),
    int(df["annual_income"].max()),
    (int(df["annual_income"].min()), int(df["annual_income"].max()))
)

filtered_df = df[
    (df["region"].isin(regions)) &
    (df["annual_income"].between(income_range[0], income_range[1]))
]

st.write("Filtered Customers:", filtered_df.shape[0])

# ---------------------------
# DATA INSPECTION
# ---------------------------
st.subheader("Data Types & Missing Values")
st.write(filtered_df.dtypes)
st.write(filtered_df.isnull().sum())

# ---------------------------
# SUMMARY STATS
# ---------------------------
st.subheader("Summary Statistics")
st.write(filtered_df.describe())

st.write("Average Purchase:", filtered_df["purchase_amount"].mean())
st.write("Median Loyalty:", filtered_df["loyalty_score"].median())

# ---------------------------
# DUPLICATES
# ---------------------------
duplicates = filtered_df.duplicated().sum()
st.write("Duplicate Rows:", duplicates)

filtered_df = filtered_df.drop_duplicates()

# ---------------------------
# REGION COUNT
# ---------------------------
st.subheader("Customers by Region")
region_counts = filtered_df["region"].value_counts()

fig, ax = plt.subplots()
sns.barplot(x=region_counts.index, y=region_counts.values, ax=ax)
st.pyplot(fig)

# ---------------------------
# AGE DISTRIBUTION
# ---------------------------
st.subheader("Age Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_df["age"], bins=15)
st.pyplot(fig)

# ---------------------------
# INCOME BOXPLOT
# ---------------------------
st.subheader("Income Boxplot")

fig, ax = plt.subplots()
sns.boxplot(x=filtered_df["annual_income"], ax=ax)
st.pyplot(fig)

# ---------------------------
# INCOME VS PURCHASE
# ---------------------------
st.subheader("Income vs Purchase Amount")

fig, ax = plt.subplots()
sns.regplot(
    x="annual_income",
    y="purchase_amount",
    data=filtered_df,
    ax=ax
)
st.pyplot(fig)

# ---------------------------
# LOYALTY VS FREQUENCY (PLOTLY)
# ---------------------------
st.subheader("Loyalty vs Purchase Frequency")

fig = px.scatter(
    filtered_df,
    x="loyalty_score",
    y="purchase_frequency",
    color="region",
    hover_data=["user_id"]
)
st.plotly_chart(fig)

# ---------------------------
# REGION-WISE AVERAGES
# ---------------------------
st.subheader("Region-wise Behavior")

region_avg = filtered_df.groupby("region")[["purchase_amount", "loyalty_score"]].mean().reset_index()

fig = px.bar(
    region_avg,
    x="region",
    y=["purchase_amount", "loyalty_score"],
    barmode="group"
)
st.plotly_chart(fig)

# ---------------------------
# AGE GROUP ANALYSIS
# ---------------------------
st.subheader("Age Group Analysis")

bins = [18, 30, 45, 60, 100]
labels = ["18-30", "31-45", "46-60", "60+"]

filtered_df["age_group"] = pd.cut(filtered_df["age"], bins=bins, labels=labels)

age_group_data = filtered_df.groupby("age_group")[["purchase_amount", "purchase_frequency"]].mean()

fig, ax = plt.subplots()
sns.heatmap(age_group_data, annot=True, fmt=".1f", ax=ax)
st.pyplot(fig)

# ---------------------------
# SCATTER MATRIX
# ---------------------------
st.subheader("Scatter Matrix")

cols = st.multiselect(
    "Select columns",
    ["age", "annual_income", "purchase_amount", "loyalty_score", "purchase_frequency"],
    default=["age", "annual_income", "purchase_amount"]
)

fig = px.scatter_matrix(filtered_df, dimensions=cols, color="region")
st.plotly_chart(fig)

# ---------------------------
# TOP CUSTOMERS
# ---------------------------
st.subheader("Top Customers")

top_n = st.selectbox("Select Top N", [5, 10, 20])

top_customers = filtered_df.sort_values(
    ["loyalty_score", "purchase_amount"],
    ascending=False
).head(top_n)

st.dataframe(top_customers)

# ---------------------------
# CORRELATION HEATMAP
# ---------------------------
st.subheader("Correlation Heatmap")

corr = filtered_df.corr(numeric_only=True)

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ---------------------------
# ADVANCED ANALYSIS (PART 5)
# ---------------------------
st.subheader("Advanced Analysis")

# High value customers
high_value = filtered_df[
    (filtered_df["purchase_amount"] >= filtered_df["purchase_amount"].quantile(0.75)) &
    (filtered_df["loyalty_score"] >= filtered_df["loyalty_score"].quantile(0.75))
]

st.write("High Value Customers:", high_value.shape[0])
st.write(high_value.describe())

# Income brackets
filtered_df["income_bracket"] = pd.cut(
    filtered_df["annual_income"],
    bins=[0, 45000, 65000, 100000],
    labels=["Low", "Medium", "High"]
)

filtered_df["spend_per_visit"] = filtered_df["purchase_amount"] / filtered_df["purchase_frequency"]

st.write(filtered_df.groupby("income_bracket")["spend_per_visit"].mean())

# ---------------------------
# BUSINESS INSIGHTS
# ---------------------------
st.subheader("📌 Business Insights & Summary")

st.markdown("""
- High-income customers tend to spend more but not always more frequently.
- Loyalty score is strongly tied to purchase frequency.
- Some regions show high income but lower loyalty → opportunity for campaigns.
- Outliers in purchase amount may skew averages.

### Recommendation:
Target medium-to-high income customers with loyalty incentives to increase repeat purchases.
""")