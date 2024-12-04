import sys
import subprocess
import streamlit as st
##v2

# Install scikit-learn if it's not already installed
try:
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn

import sklearn 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import xgboost as xgb
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

# Filter out invalid 'Exitspeed' and 'Angle' values if necessary
df = df[df['Exitspeed'].notnull()]
df = df[df['Angle'].notnull()]

# Create a mask where 'Exitspeed' and 'Angle' are not NaN
mask = (df['Exitspeed'].notnull()) & (df['Angle'].notnull())

# Streamlit Sidebar Filters
st.sidebar.header("Filter Options")

# First filter: Batter
batters = df['Batter'].dropna().unique()
batters = sorted(batters)
default_batter = batters[0] if batters else None
selected_batters = st.sidebar.multiselect(
    "Select Batter(s)",
    batters,
    default=[default_batter] if default_batter else []
)
# Add an "All" option to the list of pitcher hands
pitcher_hands = ['All', 'R', 'L']

# Allow the user to select a pitcher hand, defaulting to "All"
selected_pitcher_hand = st.sidebar.selectbox("Pitcher Hand", pitcher_hands, index=0)




# Higher-level pitch categories
pitch_categories_list = list(pitch_categories.keys())
if 'Other' in df['Pitchcategory'].unique():
    pitch_categories_list.append('Other')

selected_categories = st.sidebar.multiselect("Select Pitch Category(s)", pitch_categories_list, default=pitch_categories_list)

# Get the list of specific pitch types in the selected categories
available_pitch_types = []
for category in selected_categories:
    if category in pitch_categories:
        available_pitch_types.extend(pitch_categories[category])
    else:
        # For 'Other' category, get the pitch types not in any category
        categorized_pitches = [pitch for pitches in pitch_categories.values() for pitch in pitches]
        other_pitches = df[~df['Autopitchtype'].isin(categorized_pitches)]['Autopitchtype'].dropna()

        # Convert all entries to strings and strip whitespace
        other_pitches = other_pitches.astype(str).str.strip()

        # Exclude empty strings and 'nan' strings
        other_pitches = other_pitches[(other_pitches != '') & (other_pitches.str.lower() != 'nan')]

        available_pitch_types.extend(other_pitches.tolist())

# Convert all items to strings and strip whitespace
available_pitch_types = [str(pitch).strip() for pitch in available_pitch_types]

# Exclude empty strings and 'nan' strings
available_pitch_types = [pitch for pitch in available_pitch_types if pitch and pitch.lower() != 'nan']

# Remove duplicates
available_pitch_types = list(set(available_pitch_types))

# Sort the list
available_pitch_types = sorted(available_pitch_types)

selected_pitch_types = st.sidebar.multiselect("Select Pitch Type(s)", available_pitch_types, default=available_pitch_types)

# Map count options to boolean column names
count_option_to_column = {
    '1st-pitch': 'FirstPitch',
    '2-Strike': 'TwoStrike',
    '3-Ball': 'ThreeBall',
    'Even': 'EvenCount',
    'Hitter-Friendly': 'HitterFriendly',
    'Pitcher-Friendly': 'PitcherFriendly'
}

# Filter data based on selection
# Filter data based on selection
filtered_data = df[
    (df['Batter'].isin(selected_batters)) &
    (df['Pitchcategory'].isin(selected_categories)) &
    (df['Autopitchtype'].isin(selected_pitch_types)) &
    (df['Exitspeed'] > 0) &
    (df['Exitspeed'].notnull())
]

# Apply pitcher hand filtering if not 'All'
if selected_pitcher_hand != 'All':
    filtered_data = filtered_data[
        (filtered_data['Pitcherhand'] == selected_pitcher_hand)
    ]

def create_heatmap(data, metric, ax):
    # Check if the data is empty or the metric is not in the DataFrame
    if data.empty or metric not in data.columns:
        ax.set_title(f"No data available for {metric}.")
        ax.axis('off')
        return

    # Define the strike zone boundaries
    x_min, x_max = -2.5, 2.5
    y_min, y_max = 0, 5

    # Create 2D histogram bins
    x_bins = np.linspace(x_min, x_max, 10)
    y_bins = np.linspace(y_min, y_max, 10)

    # Compute the 2D histogram
    heatmap_data, xedges, yedges = np.histogram2d(
        data['Platelocside'],
        data['Platelocheight'],
        bins=[x_bins, y_bins],
        weights=data[metric],
        density=False
    )

    # Normalize the heatmap data
    counts, _, _ = np.histogram2d(
        data['Platelocside'],
        data['Platelocheight'],
        bins=[x_bins, y_bins]
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data = np.divide(
            heatmap_data,
            counts,
            out=np.full_like(heatmap_data, np.nan),  # Fill empty bins with NaN
            where=counts != 0
        )

    # Mask the bins with NaN
    heatmap_data = np.ma.masked_invalid(heatmap_data)

    # Set the color scale limits for the heatmap based on the metric
    if metric == 'Exitspeed':
        vmin, vmax = 60, 100
    elif metric == 'Angle':
        vmin, vmax = -45, 45
    else:
        vmin, vmax = np.nanmin(heatmap_data), np.nanmax(heatmap_data)

    # Plot the heatmap using imshow
    extent = [x_min, x_max, y_min, y_max]
    im = ax.imshow(
        heatmap_data.T,
        cmap='coolwarm',
        origin='lower',
        extent=extent,
        aspect='auto',
        vmin=vmin,
        vmax=vmax
    )

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])  # Optional: Custom ticks
    cbar.set_label(metric)  # Optional: Label the colorbar

    # Draw the strike zone rectangle
    ax.add_patch(plt.Rectangle(
        (-0.83, 1.5),
        1.66,
        2.1,
        edgecolor='black',
        facecolor='none',
        lw=2
    ))

    ax.set_title(metric)
    ax.set_xlabel('PlateLocSide')
    ax.set_ylabel('PlateLocHeight')

# Spray Chart Function
def create_spray_chart(ax):
    # Define points in polar coordinates (angle in degrees, distance in feet)
    points = [
        (-45, 90),  # Bottom left
        (-45, 315),  # Top left
        (-15, 375),  # Top center left
        (0, 405),    # Top center
        (15, 375),   # Top center right
        (45, 325),   # Top right
        (45, 90),    # Bottom right
        (0, 128),    # Bottom center (home plate extension)
        (-45, 90)    # Back to bottom left to close the shape
    ]

    # Convert polar coordinates to Cartesian coordinates
    cartesian_points = [
        (distance * np.cos(np.radians(angle)), distance * np.sin(np.radians(angle)))
        for angle, distance in points
    ]

    # Extract x and y coordinates
    x_coords, y_coords = zip(*cartesian_points)

    # Plot the lines connecting the points
    ax.plot(x_coords, y_coords, color='red', linewidth=2)

    # Set plot aesthetics
    ax.set_title("Spray Chart Outline with Foul Lines")
    ax.set_xlabel("X (Feet)")
    ax.set_ylabel("Y (Feet)")
    ax.set_xlim([-400, 400])  # Adjust as needed
    ax.set_ylim([0, 450])  # Adjust as needed
    ax.set_aspect('equal')





# Streamlit Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Heatmaps", "Spray Chart"])

# Heatmaps Page
if page == "Heatmaps":
    st.title("Hitter Heatmaps")

    # Create subplots for the heatmaps
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Heatmap for Launch Angle
    if 'Angle' in filtered_data.columns and not filtered_data['Angle'].isnull().all():
        create_heatmap(filtered_data, 'Angle', axs[0])
    else:
        axs[0].set_title("Launch Angle")
        axs[0].axis('off')
        axs[0].text(0.5, 0.5, "Launch Angle Heatmap\n(Data Not Available)", horizontalalignment='center', verticalalignment='center')

    # Heatmap for Exit Velocity
    if 'Exitspeed' in filtered_data.columns and not filtered_data['Exitspeed'].isnull().all():
        create_heatmap(filtered_data, 'Exitspeed', axs[1])
    else:
        axs[1].set_title("Exit Velocity")
        axs[1].axis('off')
        axs[1].text(0.5, 0.5, "Exit Velocity Heatmap\n(Data Not Available)", horizontalalignment='center', verticalalignment='center')

    # Heatmap for Predicted SLG (if applicable)
    axs[2].axis('off')  # Placeholder if no third heatmap is required

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

# Spray Chart Page
elif page == "Spray Chart":
    st.title("Spray Chart")
    spray_data = filtered_data[
        filtered_data['Direction'].notnull() & 
        filtered_data['Distance'].notnull()
    ]
    
    # Create Spray Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    create_spray_chart(spray_data, ax)
    st.pyplot(fig)
