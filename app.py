import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="YouTube Subscriber Predictor", layout="wide")

st.title("📈 YouTube Subscriber Growth Predictor")
st.markdown("Predict future subscriber growth using a Logistic Growth Model.")

# ----------------------------
# Sidebar - User Inputs
# ----------------------------
st.sidebar.header("🔧 Input Parameters")

# Suggestions for inputs
st.sidebar.markdown("### 💡 Suggested Input Ranges")
st.sidebar.info("""
- Initial Subscribers: 100 – 10,000  
- Max Audience (K): 10,000 – 10,000,000  
- Growth Rate (r): 0.01 – 1.0  
- Duration: 12 – 60 months  
""")

S0 = st.sidebar.number_input("Initial Subscribers (S0)", min_value=1, value=1000)
K = st.sidebar.number_input("Maximum Audience Size (K)", min_value=S0+1, value=100000)
r = st.sidebar.slider("Growth Rate (r)", min_value=0.01, max_value=1.0, value=0.2)
months = st.sidebar.slider("Prediction Duration (Months)", 1, 120, 36)

# ----------------------------
# Logistic Growth Function
# ----------------------------
def logistic_growth(t, S0, K, r):
    return K / (1 + ((K - S0) / S0) * np.exp(-r * t))

# Time array
t = np.arange(0, months + 1)

# Prediction
subscribers = logistic_growth(t, S0, K, r)

# Create DataFrame
df = pd.DataFrame({
    "Month": t,
    "Predicted Subscribers": subscribers.astype(int)
})

# ----------------------------
# Main Visualization
# ----------------------------
st.subheader("📊 Growth Prediction Chart")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df["Month"], df["Predicted Subscribers"], linewidth=3)
ax.set_xlabel("Months")
ax.set_ylabel("Subscribers")
ax.set_title("Predicted Subscriber Growth")
ax.grid(True)

st.pyplot(fig)

# ----------------------------
# Milestone Chart
# ----------------------------
st.subheader("🏆 Milestone Achievement")

milestones = [1000, 10000, 100000, 1000000, 10000000]

milestone_data = []

for milestone in milestones:
    reached = df[df["Predicted Subscribers"] >= milestone]
    if not reached.empty:
        month_reached = reached.iloc[0]["Month"]
        milestone_data.append((milestone, int(month_reached)))
    else:
        milestone_data.append((milestone, "Not reached"))

milestone_df = pd.DataFrame(milestone_data, columns=["Milestone Subscribers", "Month Reached"])

st.table(milestone_df)

# ----------------------------
# Final Stats
# ----------------------------
st.subheader("📌 Summary Statistics")

final_subs = int(subscribers[-1])
growth_percent = ((final_subs - S0) / S0) * 100

col1, col2 = st.columns(2)

col1.metric("Final Predicted Subscribers", f"{final_subs:,}")
col2.metric("Total Growth %", f"{growth_percent:.2f}%")

st.success("Prediction Complete! Adjust parameters in the sidebar to explore different growth scenarios.")
