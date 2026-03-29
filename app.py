import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="YouTube Subscriber Predictor", layout="wide")

# ----------------------------
# Title
# ----------------------------
st.title("📈 YouTube Subscriber Growth Predictor")
st.markdown("Predict future subscriber growth using a Logistic Growth Model.")

# ----------------------------
# Logistic Formula Section
# ----------------------------
st.subheader("📘 Logistic Growth Model")

st.latex(r"N(t) = \frac{K}{1 + \frac{K - S_0}{S_0} e^{-rt}}")

st.markdown("""
Where:
- **K** = Maximum audience (carrying capacity)  
- **S₀** = Initial subscribers  
- **r** = Growth rate  
- **t** = Time (months)  

👉 Growth starts slow → increases rapidly → slows near saturation.
""")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🔧 Input Parameters")

st.sidebar.info("""
Suggested:
- Initial Subscribers: 100 – 10,000  
- Max Audience (K): 10,000 – 10,000,000  
- Growth Rate: 0.01 – 1.0  
""")

S0 = st.sidebar.number_input("Initial Subscribers (S0)", min_value=1, value=1000)
K = st.sidebar.number_input("Maximum Audience Size (K)", min_value=S0+1, value=100000)
r = st.sidebar.slider("Growth Rate (r)", 0.01, 1.0, 0.2)
months = st.sidebar.slider("Prediction Duration (Months)", 1, 120, 36)

target = st.sidebar.number_input("🎯 Target Subscribers", value=50000)

viral = st.sidebar.checkbox("🚀 Enable Viral Growth Spike")

# ----------------------------
# Logistic Function
# ----------------------------
def logistic_growth(t, S0, K, r):
    return K / (1 + ((K - S0) / S0) * np.exp(-r * t))

t = np.arange(0, months + 1)
subscribers = logistic_growth(t, S0, K, r)

# Viral Effect
if viral:
    noise = np.random.normal(0, 0.05*K, len(subscribers))
    subscribers = np.clip(subscribers + noise, 0, None)

# DataFrame
df = pd.DataFrame({
    "Month": t,
    "Predicted Subscribers": subscribers.astype(int)
})

# ----------------------------
# Graph
# ----------------------------
st.subheader("📊 Growth Prediction Chart")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(df["Month"], df["Predicted Subscribers"], linewidth=3, label="Predicted Growth")

# Growth phases
ax.axvspan(0, months*0.3, alpha=0.1, label="Slow Growth")
ax.axvspan(months*0.3, months*0.7, alpha=0.2, label="Rapid Growth")
ax.axvspan(months*0.7, months, alpha=0.1, label="Saturation")

# Milestones
milestones = [1000, 10000, 100000, 1000000, 10000000]
for m in milestones:
    ax.axhline(y=m, linestyle='--')

# CSV Upload for real data
uploaded_file = st.file_uploader("Upload past subscriber data (CSV)", type=["csv"])

if uploaded_file:
    real_df = pd.read_csv(uploaded_file)
    ax.plot(real_df["Month"], real_df["Subscribers"], linestyle='dashed', label="Actual Data")

ax.set_xlabel("Months")
ax.set_ylabel("Subscribers")
ax.set_title("Subscriber Growth Prediction")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ----------------------------
# Target Prediction
# ----------------------------
target_reached = df[df["Predicted Subscribers"] >= target]

if not target_reached.empty:
    st.success(f"🎯 You will reach {target:,} subscribers in month {int(target_reached.iloc[0]['Month'])}")
else:
    st.warning("Target not reached in selected duration.")

# ----------------------------
# Milestones Table
# ----------------------------
st.subheader("🏆 Milestone Achievement")

milestone_data = []

for milestone in milestones:
    reached = df[df["Predicted Subscribers"] >= milestone]
    if not reached.empty:
        milestone_data.append((milestone, int(reached.iloc[0]["Month"])))
    else:
        milestone_data.append((milestone, "Not reached"))

milestone_df = pd.DataFrame(milestone_data, columns=["Milestone Subscribers", "Month Reached"])
st.table(milestone_df)

# ----------------------------
# Monthly Growth Chart
# ----------------------------
st.subheader("📉 Monthly Growth Trend")

df["Monthly Growth"] = df["Predicted Subscribers"].diff()
st.line_chart(df["Monthly Growth"])

# ----------------------------
# Summary
# ----------------------------
st.subheader("📌 Summary Statistics")

final_subs = int(subscribers[-1])
growth_percent = ((final_subs - S0) / S0) * 100

col1, col2 = st.columns(2)

col1.metric("Final Predicted Subscribers", f"{final_subs:,}")
col2.metric("Total Growth %", f"{growth_percent:.2f}%")

# ----------------------------
# Limitations
# ----------------------------
st.subheader("⚠️ Model Limitations")

st.markdown("""
- Assumes smooth growth (real YouTube growth may have spikes)  
- Does not include algorithm changes  
- Ignores content quality and engagement factors  
- Viral growth is simulated randomly  
""")

st.success("✅ Prediction Complete! Try different scenarios using sidebar.")
