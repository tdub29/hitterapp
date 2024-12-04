import streamlit as st
import sys
import subprocess
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##v2

# Install scikit-learn if it's not already installed
try:
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn

# Load required libraries
import sklearn
from statsmodels.nonparametric.kernel_regression import KernelReg

# Define custom color palette
kde_min = '#236abe'
kde_mid = '#fefefe'
kde_max = '#a9373b'

kde_palette = (sns.color_palette(f'blend:{kde_min},{kde_mid}', n_colors=1001)[:-1] +
               sns.color_palette(f'blend:{kde_mid},{kde_max}', n_colors=1001)[:-1])

## Set Styling
# Plot Style
pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

sns.set_theme(
    style={
        'axes.edgecolor': pl_white,
        'axes.facecolor': pl_white,
        'axes.labelcolor': pl_white,
        'xtick.color': pl_white,
        'ytick.color': pl_white,
        'figure.facecolor': pl_background,
        'grid.color': pl_background,
        'grid.linestyle': '-',
        'legend.facecolor': pl_background,
        'text.color': pl_white
     }
    )

# Load the CSV file
file_path = "https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/usd_baseball_TM_master_file.csv"
df = pd.read_csv(file_path)

# Standardize column capitalization
df.columns = [col.strip().capitalize() for col in df.columns]

# Ensure 'Balls' and 'Strikes' columns are numeric
df['Balls'] = pd.to_numeric(df['Balls'], errors='coerce')
df['Strikes'] = pd.to_numeric(df['Strikes'], errors='coerce')

# Add a 'PitcherHand' column based on 'Relside'
def determine_pitcher_hand(rel_side):
    if rel_side > 0:
        return 'R'
    elif rel_side < 0:
        return 'L'
    else:
        return 'Unknown'

df['Pitcherhand'] = df['Relside'].apply(determine_pitcher_hand)

# Define pitch categories based on initial pitch types
pitch_categories = {
    "Breaking Ball": ["Slider", "Curveball"],
    "Fastball": ["Fastball", "Four-Seam", "Sinker", "Cutter"],
    "Offspeed": ["ChangeUp", "Splitter"]
}

# Function to categorize pitch types into broader groups
def categorize_pitch_type(pitch_type):
    for category, pitches in pitch_categories.items():
        if pitch_type in pitches:
            return category
    return "Other"

# Create a new column 'Pitchcategory' to categorize pitches
df['Pitchcategory'] = df['Autopitchtype'].apply(categorize_pitch_type)

# Create boolean columns for each count category
df['FirstPitch'] = (df['Balls'] == 0) & (df['Strikes'] == 0)
df['TwoStrike'] = df['Strikes'] == 2
df['ThreeBall'] = df['Balls'] == 3
df['EvenCount'] = (df['Balls'] == df['Strikes']) & (df['Balls'] != 0)
df['HitterFriendly'] = df['Balls'] > df['Strikes']
df['PitcherFriendly'] = df['Strikes'] > df['Balls']

# Ensure 'Exitspeed' and 'Angle' are numeric
df['Exitspeed'] = pd.to_numeric(df['Exitspeed'], errors='coerce')
df['Angle'] = pd.to_numeric(df['Angle'], errors='coerce')

# Filter data based on valid 'Exitspeed' and 'Angle'
df = df[df['Exitspeed'].notnull() & df['Angle'].notnull()]

# Sidebar Filters
st.sidebar.header("Filter Options")
pitcher_hands = ['All', 'R', 'L']
selected_pitcher_hand = st.sidebar.selectbox("Pitcher Hand", pitcher_hands, index=0)

# Data Filtering
filtered_data = df.copy()
if selected_pitcher_hand != 'All':
    filtered_data = filtered_data[filtered_data['Pitcherhand'] == selected_pitcher_hand]

# Define Heatmap Function
def create_heatmap(data, metric, ax):
    if data.empty or metric not in data.columns:
        ax.set_title(f"No data available for {metric}.")
        ax.axis('off')
        return
    x_min, x_max = -2.5, 2.5
    y_min, y_max = 0, 5
    x_bins = np.linspace(x_min, x_max, 20)
    y_bins = np.linspace(y_min, y_max, 20)
    heatmap_data, _, _ = np.histogram2d(data['Platelocside'], data['Platelocheight'], bins=[x_bins, y_bins], weights=data[metric])
    counts, _, _ = np.histogram2d(data['Platelocside'], data['Platelocheight'], bins=[x_bins, y_bins])
    heatmap_data = np.divide(heatmap_data, counts, out=np.zeros_like(heatmap_data), where=counts != 0)
    im = ax.imshow(heatmap_data.T, cmap='coolwarm', extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax).set_label(metric)

# Define Spray Chart Function
def create_spray_chart(data, ax):
    data['Direction_rad'] = np.radians(data['Direction'])
    data['X'] = data['Distance'] * np.cos(data['Direction_rad'])
    data['Y'] = data['Distance'] * np.sin(data['Direction_rad'])
    scatter = ax.scatter(data['X'], data['Y'], c=data['Exitspeed'], cmap='coolwarm', s=50, edgecolor='k')
    plt.colorbar(scatter, ax=ax).set_label('Exit Speed')
    ax.set_title("Spray Chart")
    ax.set_aspect('equal')

# Main App Logic
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Heatmaps", "Spray Chart"])

if page == "Heatmaps":
    st.title("Hitter Heatmaps")
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    create_heatmap(filtered_data, 'Angle', axs[0])
    create_heatmap(filtered_data, 'Exitspeed', axs[1])
    st.pyplot(fig)
elif page == "Spray Chart":
    st.title("Spray Chart")
    spray_data = filtered_data[filtered_data['Direction'].notnull() & filtered_data['Distance'].notnull()]
    fig, ax = plt.subplots(figsize=(10, 8))
    create_spray_chart(spray_data, ax)
    st.pyplot(fig)
