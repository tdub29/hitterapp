import sys
import subprocess
import streamlit as st
##v2

# Install scikit-learn if it's not already installed
# Upgrade to compatible versions of scikit-learn and xgboost
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
from matplotlib.patches import Polygon, Rectangle
import matplotlib.lines as mlines

booster = xgb.Booster()
booster.load_model('xSLG_model.json')

best_model = xgb.XGBRegressor()
best_model._Booster = booster  # Assign the booster to the regressor

# 1) Load the No-Swing model (JSON)
no_swing_booster = xgb.Booster()
no_swing_booster.load_model('model_no_swing.json')
model_no_swing = xgb.XGBRegressor()
model_no_swing._Booster = no_swing_booster

# 2) Load the Swing model (JSON)
swing_booster = xgb.Booster()
swing_booster.load_model('model_swing.json')
model_swing = xgb.XGBRegressor()
model_swing._Booster = swing_booster


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

# Ensure 'Balls' and 'strikes' columns are numeric
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
df['Firstpitch'] = (df['Balls'] == 0) & (df['Strikes'] == 0)
df['Twostrike'] = df['Strikes'] == 2
df['Threeball'] = df['Balls'] == 3
df['Evencount'] = (df['Balls'] == df['Strikes']) & (df['Balls'] != 0)
df['Hitterfriendly'] = df['Balls'] > df['Strikes']
df['Pitcherfriendly'] = df['Strikes'] > df['Balls']

# Ensure 'Exitspeed' and 'Angle' are numeric
df['Exitspeed'] = pd.to_numeric(df['Exitspeed'], errors='coerce')
df['Angle'] = pd.to_numeric(df['Angle'], errors='coerce')

# Filter out invalid 'Exitspeed' and 'Angle' values if necessary
# df = df[df['Exitspeed'].notnull()]
# df = df[df['Angle'].notnull()]

# Create a mask where 'Exitspeed' and 'Angle' are not NaN
# mask = (df['Exitspeed'].notnull()) & (df['Angle'].notnull())

# Streamlit Sidebar Filters
st.sidebar.header("Filter Options")

# First filter: Batter
batters = df['Batter'].dropna().unique()
batters = sorted(batters)
batters = ["All Hitters"] + list(batters)  # Add "All Hitters" option at the front
default_batter = ["All Hitters"] if "All Hitters" in batters else [batters[0]] if batters else []

selected_batters = st.sidebar.multiselect(
    "Select Batter(s)",
    batters,
    default=default_batter
)

# If "All Hitters" is selected, we consider all batters
if "All Hitters" in selected_batters:
    batter_filter = True  # No restriction on batters
else:
    batter_filter = df['Batter'].isin(selected_batters)





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
        other_pitches = other_pitches[(other_pitches != '') & (other_pitches.str.lower() != 'nan')]

        available_pitch_types.extend(other_pitches.tolist())

available_pitch_types = [str(pitch).strip() for pitch in available_pitch_types]
available_pitch_types = [pitch for pitch in available_pitch_types if pitch and pitch.lower() != 'nan']
available_pitch_types = list(set(available_pitch_types))
available_pitch_types = sorted(available_pitch_types)

selected_pitch_types = st.sidebar.multiselect("Select Pitch Type(s)", available_pitch_types, default=available_pitch_types)

# Map count options to boolean column names
count_option_to_column = {
    '1st-pitch': 'Firstpitch',
    '2-Strike': 'Twostrike',
    '3-Ball': 'Threeball',
    'Even': 'Evencount',
    'Hitter-Friendly': 'Hitterfriendly',
    'Pitcher-Friendly': 'Pitcherfriendly'
}

df['Event'] = np.where(
    df['Playresult'].isin(['Undefined', 'StolenBase', 'CaughtStealing']),
    np.where(
        df['Pitchcall'].eq('BallInDirt'),
        'BallCalled',
        np.where(
            df['Pitchcall'].str.contains('Foul', case=False, na=False),
            'Foul',
            df['Pitchcall']
        )
    ),
    df['Playresult']
)


# Create 'Swing' column based on Exitspeed and Pitchcall
df['Swing'] = np.where(
    (df['Exitspeed'] > 0) | (df['Pitchcall'].str.contains('Swing|Foul', case=False, na=False)),
    'Swing',  # Label as Swing if Exitspeed > 0 or Pitchcall contains 'Swing'/'Foul'
    'Take'    # Otherwise, label as Take
)

# Define strike zone boundaries
STRIKE_ZONE_SIDE = (-0.83, 0.83)  # Plate width in feet
STRIKE_ZONE_HEIGHT = (1.5, 3.5)   # Typical strike zone height in feet

# Create 'Zone' column based on Platelocside and Platelocheight
df['Zone'] = np.where(
    (df['Platelocside'].between(*STRIKE_ZONE_SIDE)) & 
    (df['Platelocheight'].between(*STRIKE_ZONE_HEIGHT)),
    'InZone',  # If pitch is within both side and height boundaries
    'Out'      # Otherwise, classify as Out
)

# Create 'Atbatid' by converting and concatenating specific columns without modifying 'Date'
df['Atbatid'] = (
    df['Date'].astype(str).fillna('') + '_' +
    df['Pitcher'].fillna('').astype(str) + '_' +
    df['Paofinning'].fillna('').astype(str) + '_' +
    df['Inning'].fillna('').astype(str)
)

# Create 'Contact' column based on Exitspeed
df['Contact'] = np.where(df['Exitspeed'] > 0, 'Yes', 'No')

df['Whiff'] = np.where(
    (df['Swing'] == 'Swing') & (df['Contact'] == 'No'),
    'Yes',  # Mark as 'Yes' if both conditions are met
    'No'    # Otherwise, mark as 'No'
)

# Ensure Balls and Strikes are numeric
df['Balls'] = pd.to_numeric(df['Balls'], errors='coerce').fillna(0).astype(int)
df['Strikes'] = pd.to_numeric(df['Strikes'], errors='coerce').fillna(0).astype(int)

# Create 'Count' column as a string combining Balls and Strikes
df['Count'] = df['Balls'].astype(str) + '-' + df['Strikes'].astype(str)


# Map Plate Zones based on PlatelocSide and PlatelocHeight
def map_plate_zone(row):
    """
    Map pitch location to plate zones (Heart, Shadow, Chase, Waste) based on Platelocside and Platelocheight in feet.
    """
    side = row['Platelocside']
    height = row['Platelocheight']

    # Convert inches to feet for thresholds
    HEART_SIDE = (-0.56, 0.56)  # 6.7 inches → ~0.56 feet
    HEART_HEIGHT = (1.83, 3.17)  # 22-38 inches → 1.83-3.17 feet

    SHADOW_SIDE = (-1.11, 1.11)  # 13.3 inches → ~1.11 feet
    SHADOW_HEIGHT = (1.17, 3.83)  # 14-46 inches → 1.17-3.83 feet

    CHASE_SIDE = (-1.67, 1.67)  # 20 inches → ~1.67 feet
    CHASE_HEIGHT = (0.5, 4.33)  # 6-52 inches → 0.5-4.33 feet

    WASTE_SIDE = (-2.5, 2.5)  # Beyond Chase Zone (arbitrary large range)
    WASTE_HEIGHT = (0.0, 5.0)  # Below 6 inches (0.5 feet) or above 52 inches (4.33 feet)

    # Map to zones
    if HEART_SIDE[0] <= side <= HEART_SIDE[1] and HEART_HEIGHT[0] <= height <= HEART_HEIGHT[1]:
        return 'Heart'
    elif SHADOW_SIDE[0] <= side <= SHADOW_SIDE[1] and SHADOW_HEIGHT[0] <= height <= SHADOW_HEIGHT[1]:
        return 'Shadow'
    elif CHASE_SIDE[0] <= side <= CHASE_SIDE[1] and CHASE_HEIGHT[0] <= height <= CHASE_HEIGHT[1]:
        return 'Chase'
    else:
        return 'Waste'


# Apply the mapping function to the dataframe
df['PlateZone'] = df.apply(map_plate_zone, axis=1)

# First, rename the columns
df.rename(columns={'Strikes': 'strikes', 'Balls': 'balls'}, inplace=True)


# 1) Split
df_no_swing = df[df['Swing'] == 'Take'].dropna(
    subset=['Platelocside','Platelocheight','strikes','balls']
).copy()

df_swing = df[df['Swing'] == 'Swing'].dropna(
    subset=['Platelocside','Platelocheight','strikes','balls']
).copy()

# 3) Predict with each model using the DataFrame
df_no_swing['decision_rv'] = model_no_swing.predict(
    df_no_swing[['Platelocside','Platelocheight','strikes','balls']]
)
df_swing['decision_rv'] = model_swing.predict(
    df_swing[['Platelocside','Platelocheight','strikes','balls']]
)

# 4) Merge predictions back to the main df
#    If you have a unique pitch identifier (e.g. 'Pitchuid'), merge on that:
#    We'll do a left merge so all rows remain, even if no predictions are found.
if 'Pitchuid' in df.columns:
    # Extract only the columns we need for merging
    df_no_swing_merge = df_no_swing[['Pitchuid','decision_rv']]
    df_swing_merge    = df_swing[['Pitchuid','decision_rv']]

    # Rename to avoid overwriting
    df_no_swing_merge = df_no_swing_merge.rename(columns={'decision_rv': 'decision_rv_no_swing'})
    df_swing_merge    = df_swing_merge.rename(columns={'decision_rv': 'decision_rv_swing'})

    df = df.merge(df_no_swing_merge, on='Pitchuid', how='left')
    df = df.merge(df_swing_merge, on='Pitchuid', how='left')

    # Combine them into a single 'decision_rv' column if desired:
    # e.g. if row is no-swing, 'decision_rv_no_swing' is valid, else 'decision_rv_swing'
    # We'll just fill one from the other:
    df['decision_rv'] = df['decision_rv_no_swing'].combine_first(df['decision_rv_swing'])

else:
    # If no unique ID, you might do a concat back approach. That is trickier
    # because you must align on index or multi-column merges. For now, we assume 'Pitchuid' exists.
    pass





# ADD PITCHER FILTER HERE
df['Pitcher'].fillna('Machine', inplace=True)
pitchers = df['Pitcher'].unique()
pitchers = sorted(pitchers)

# By default, select all pitchers
selected_pitchers = st.sidebar.multiselect(
    "Select Pitcher(s)",
    pitchers,
    default=pitchers  # all selected by default
)

# If no pitchers are selected, filtered_data would be empty, so you may want to handle that case.
# For now, if user deselects all, it means no pitcher selected (filtered_data will reflect that).
pitcher_filter = df['Pitcher'].isin(selected_pitchers)

# Apply filters using batter_filter (without Exitspeed filter)
all_pitches = df[
    (batter_filter) &
    (pitcher_filter) &
    (df['Pitchcategory'].isin(selected_categories)) &
    (df['Autopitchtype'].isin(selected_pitch_types)) 
]

# Apply pitcher hand filtering if not 'All'
if selected_pitcher_hand != 'All':
    all_pitches = all_pitches[
        (all_pitches['Pitcherhand'] == selected_pitcher_hand)
    ]

# Apply additional Exitspeed filter for filtered_data
filtered_data = all_pitches[
    (all_pitches['Exitspeed'] > 0) &
    (all_pitches['Exitspeed'].notnull())  # Ensure valid Exitspeed data, but don't filter on > 0
]


# Apply pitcher hand filtering if not 'All'
if selected_pitcher_hand != 'All':
    filtered_data = filtered_data[(filtered_data['Pitcherhand'] == selected_pitcher_hand)]

# Create 'interaction' for xSLG predictions
filtered_data['interaction'] = filtered_data['Exitspeed'] * filtered_data['Angle']

# Predict xSLG
# Create the prediction DataFrame with correct column names
X_pred = filtered_data[['Exitspeed', 'Angle', 'interaction']].copy()
X_pred.rename(columns={'Exitspeed':'launch_speed', 'Angle':'launch_angle'}, inplace=True)

# Now predict using the corrected feature names
filtered_data['xSLG'] = best_model.predict(X_pred)

# Ensure Pitchuid exists in both datasets
if 'Pitchuid' in filtered_data.columns and 'Pitchuid' in all_pitches.columns:
    # Select relevant columns from filtered_data
    xSLG_data = filtered_data[['Pitchuid', 'xSLG']].copy()
    
    # Drop duplicates to avoid ambiguity in join
    xSLG_data = xSLG_data.drop_duplicates(subset='Pitchuid')
    
    # Perform the join on Pitchuid
    all_pitches = all_pitches.merge(xSLG_data, on='Pitchuid', how='left')
    
    # Fill any missing xSLG values with NaN
    all_pitches['xSLG'] = all_pitches['xSLG'].fillna(np.nan)


def create_heatmap(data, metric, ax):
    if data.empty or metric not in data.columns:
        ax.set_title(f"No data available for {metric}.")
        ax.axis('off')
        return

    x_min, x_max = -2.5, 2.5
    y_min, y_max = 0, 5

    x_bins = np.linspace(x_min, x_max, 10)
    y_bins = np.linspace(y_min, y_max, 10)

    heatmap_data, xedges, yedges = np.histogram2d(
        data['Platelocside'],
        data['Platelocheight'],
        bins=[x_bins, y_bins],
        weights=data[metric],
        density=False
    )

    counts, _, _ = np.histogram2d(
        data['Platelocside'],
        data['Platelocheight'],
        bins=[x_bins, y_bins]
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data = np.divide(
            heatmap_data,
            counts,
            out=np.full_like(heatmap_data, np.nan),
            where=counts != 0
        )

    heatmap_data = np.ma.masked_invalid(heatmap_data)

    # Handle case where no valid data for xSLG exists
    if metric == 'xSLG' and np.isnan(heatmap_data).all():
        ax.set_title("No data available for xSLG.")
        ax.axis('off')
        return

    if metric == 'Exitspeed':
        vmin, vmax = 60, 100
    elif metric == 'Angle':
        vmin, vmax = -45, 45
    elif metric == 'xSLG':
        # Use a fixed range for xSLG
        vmin, vmax = 0.25, 0.65
    else:
        vmin, vmax = np.nanmin(heatmap_data), np.nanmax(heatmap_data)

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

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)

    ax.add_patch(plt.Rectangle(
        (-0.83, 1.5),
        1.66,
        2.1,
        edgecolor='black',
        facecolor='none',
        lw=2
    ))

     # Add plate polygon
    plate_vertices = [(-0.83, 0.1), (0.83, 0.1), (0.65, 0.25), (0, 0.5), (-0.65, 0.25)]
    plate = plt.Polygon(plate_vertices, closed=True, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(plate)

    ax.set_title(metric, fontsize=20)
    ax.set_xlabel('PlateLocSide')
    ax.set_ylabel('PlateLocHeight')

def plot_pitch_locations_by_playresult(data):
    if data.empty:
        st.warning("No data available for the selected filters to plot pitch locations.")
        return

    # Ensure Swing and xSLG fields exist and are properly formatted
    data['Swing'] = data['Swing'].astype('category')
    data['xSLG'] = pd.to_numeric(data['xSLG'].fillna(0), errors='coerce').fillna(0)
    data['Count'] = data['balls'].astype(str) + '-' + data['strikes'].astype(str)
    
    # Define pitcher sides and swing types
    pitcher_sides = ['R', 'L']
    swing_types = ['Swing', 'Take']
    plate_vertices = [(-0.83, 0.1), (0.83, 0.1), (0.65, 0.25), (0, 0.5), (-0.65, 0.25)]

    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 16), sharey=True, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    axes = axes.flatten()  # Flatten to simplify indexing
    
    # Define colormap and normalization for xSLG
    norm = plt.Normalize(vmin=0, vmax=0.8)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])  # Required for colorbar

    # Zone Definitions
    zone_definitions = {
        'Heart': {'x_range': (-0.56, 0.56), 'y_range': (1.83, 3.17), 'color': '#7CFC00', 'alpha': 0.3},
        'Shadow': {'x_range': (-1.11, 1.11), 'y_range': (1.17, 3.83), 'color': '#FFD700', 'alpha': 0.2},
        'Chase': {'x_range': (-1.67, 1.67), 'y_range': (0.5, 4.33), 'color': '#FFA07A', 'alpha': 0.15},
        'Waste': {'x_range': (-2.5, 2.5), 'y_range': (0.0, 5.0), 'color': '#FF6347', 'alpha': 0.1}
    }

    # Loop through Swing/Take and R/L hand combinations
    for i, (swing, pitcher_side) in enumerate([(s, p) for s in swing_types for p in pitcher_sides]):
        side_data = data[(data['Swing'] == swing) & (data['Pitcherhand'] == pitcher_side)]
        
        for _, row in side_data.iterrows():
            marker = 'o' if row['Whiff'] == 'No' else 'x'
            axes[i].scatter(
                row['Platelocside'],
                row['Platelocheight'],
                c=row['xSLG'],
                cmap='coolwarm',
                norm=norm,
                edgecolor='black',
                s=100,
                marker=marker,
                zorder=3
            )
            
            # Add Count Labels for 'Take' Plots
            if swing == 'Take':
                axes[i].text(
                    row['Platelocside'] - 0.15,
                    row['Platelocheight'],
                    row['Count'],
                    fontsize=8,
                    color='darkblue',
                    ha='right',
                    va='center',
                    zorder=4
                )

        # Add Zone Shading
        for zone, props in zone_definitions.items():
            axes[i].add_patch(Rectangle(
                (props['x_range'][0], props['y_range'][0]),
                props['x_range'][1] - props['x_range'][0],
                props['y_range'][1] - props['y_range'][0],
                edgecolor='none',
                facecolor=props['color'],
                alpha=props['alpha'],
                zorder=0  # Ensure shading is below scatter points
            ))

        # Add strike zone rectangle
        axes[i].add_patch(Rectangle(
            (-0.83, 1.5),
            1.66,
            2.1,
            edgecolor='black',
            facecolor='none',
            lw=2
        ))
        
        # Add home plate polygon
        plate = Polygon(plate_vertices, closed=True, linewidth=1, edgecolor='k', facecolor='none')
        axes[i].add_patch(plate)
        
        # Titles for each subplot
        axes[i].set_title(f'{swing} vs {pitcher_side}-Handed Pitchers')
        axes[i].set_xlim(-2.5, 2.5)
        axes[i].set_ylim(0, 5)
        axes[i].set_xlabel('PlateLocSide')
        axes[i].set_ylabel('PlateLocHeight')

        # Remove any auto-generated legend
        if axes[i].get_legend() is not None:
            axes[i].get_legend().remove()

    # Add a single colorbar legend for xSLG across all plots
    cbar_ax = fig.add_axes([0.3, 0.5, 0.4, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    # Customize colorbar appearance
    cbar.set_label('xSLG (0 - 0.8)', fontsize=12, color='white')
    cbar.ax.tick_params(labelcolor='white')
    
    # Adjust spacing between rows and columns
    plt.subplots_adjust(hspace=0.4)  # More space vertically
    
    st.pyplot(fig)

def plot_pitch_locations_by_hand_and_ypred(data):
    """
    Plot pitch locations vs. left- and right-handed pitchers,
    colored by decision_rv, shaped by Event.
    Adds zone shading, strike zone rectangle,
    and an inset plot in the bottom-right corner.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon

    if data.empty:
        st.warning("No data available for the selected filters.")
        return

    # Marker shapes for final Event values
    shape_map = {
        'BallCalled': 'o',
        'Foul': '^',
        'StrikeCalled': 's',
        'StrikeSwinging': 'D',
        'HitByPitch': 'p',
        'Single': 'v',
        'Double': '>',
        'Triple': '<',
        'HomeRun': '*',
        'Out': 'X',
        'Undefined': '8'
    }

    # Zone definitions
    zone_definitions = {
        'Heart':  {'x_range': (-0.56, 0.56), 'y_range': (1.83, 3.17), 'color': '#7CFC00', 'alpha': 0.3},
        'Shadow': {'x_range': (-1.11, 1.11), 'y_range': (1.17, 3.83), 'color': '#FFD700', 'alpha': 0.2},
        'Chase':  {'x_range': (-1.67, 1.67), 'y_range': (0.5, 4.33),  'color': '#FFA07A', 'alpha': 0.15},
        'Waste':  {'x_range': (-2.5, 2.5),   'y_range': (0.0, 5.0),   'color': '#FF6347', 'alpha': 0.1}
    }

    pitcher_sides = ['R', 'L']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    for i, side in enumerate(pitcher_sides):
        subset = data[data['Pitcherhand'] == side]

        # 1) Plot each Event with a specific marker
        for event_type in subset['Event'].unique():
            sub_e = subset[subset['Event'] == event_type]
            marker_style = shape_map.get(event_type, 'o')  # default circle
            sc = axes[i].scatter(
                sub_e['Platelocside'],
                sub_e['Platelocheight'],
                c=sub_e['decision_rv'],
                cmap='coolwarm',
                marker=marker_style,
                edgecolor='black',
                vmin=-0.2,   # adjust color scale as needed
                vmax=0.2,
                s=80,
                alpha=0.7
            )

        # 2) Add zone shading (Heart/Shadow/Chase/Waste)
        for zone, props in zone_definitions.items():
            axes[i].add_patch(Rectangle(
                (props['x_range'][0], props['y_range'][0]),
                props['x_range'][1] - props['x_range'][0],
                props['y_range'][1] - props['y_range'][0],
                edgecolor='none',
                facecolor=props['color'],
                alpha=props['alpha'],
                zorder=0
            ))

        # 3) Add the strike zone rectangle
        axes[i].add_patch(Rectangle(
            (-0.83, 1.5),
            1.66,   # width
            2.0,    # height
            edgecolor='black',
            facecolor='none',
            lw=2
        ))

        axes[i].set_title(f"Pitcher Side: {side}")
        axes[i].set_xlim(-2.5, 2.5)
        axes[i].set_ylim(0, 5)
        axes[i].set_xlabel("PlateLocSide")
        axes[i].set_ylabel("PlateLocHeight")

    # 4) Add a shared colorbar for decision_rv
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist())
    cbar.set_label("decision_rv", fontsize=12)

    # 5) Add a small inset plot in the bottom-right corner (below legend)
    handles = []
    for event_name, marker_shape in shape_map.items():
        # Create an invisible Line2D with the right marker and label
        handle = mlines.Line2D(
            [], [], 
            color='black', 
            marker=marker_shape, 
            markersize=8, 
            label=event_name, 
            linestyle='None'  # no connecting line
        )
        handles.append(handle)
    
    # Add the legend to the bottom-right corner of the figure
    legend_obj = fig.legend(
        handles=handles,
        loc='lower right',  # bottom right of the entire figure
        frameon=True,
        title='Shapes'
    )
    
    # Set the title color
    legend_obj.get_title().set_color('darkblue')
    
    # Set the color for all legend labels
    for text_obj in legend_obj.get_texts():
        text_obj.set_color('darkblue')


    st.pyplot(fig)







def create_spray_chart(data, ax):
    outline_points = [
        (0, 0),
        (-45, 90),
        (-45, 315),
        (-15, 375),
        (0, 405),
        (15, 375),
        (45, 325),
        (45, 90),
        (0, 128),
        (-45, 90),
        (0, 0)
    ]

    outline_cartesian = [
        (-distance * np.sin(np.radians(angle)), distance * np.cos(np.radians(angle)))
        for angle, distance in outline_points
    ]

    outline_x, outline_y = zip(*outline_cartesian)
    ax.plot(outline_x, outline_y, color='red', linewidth=2, label="Field Outline")

    if 'Direction' in data.columns and 'Distance' in data.columns:
        data['Direction_rad'] = np.radians(data['Direction'])
        data['Rotated_X'] = -data['Distance'] * np.sin(data['Direction_rad'])
        data['Rotated_Y'] = data['Distance'] * np.cos(data['Direction_rad'])

        scatter = ax.scatter(
            data['Rotated_X'],
            data['Rotated_Y'],
            c=data['Exitspeed'],
            cmap='coolwarm',
            s=50,
            edgecolor='k',
            label='Hits',
            vmin=60,  # Set minimum value for color bar
            vmax=100  # Set maximum value for color bar
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Exit Speed")

    ax.set_title("Spray Chart")
    ax.set_xlabel("X (Feet)")
    ax.set_ylabel("Y (Feet)")
    ax.set_xlim([-250, 250])
    ax.set_ylim([0, 450])
    ax.set_aspect('equal')
    ax.legend()

def display_hitter_metrics(all_pitches):
    """
    Displays hitter metrics in a tabular format on a Streamlit app,
    AND includes No-Swing, Swing, and Overall decision values (20–80).
    """
    if all_pitches.empty:
        st.write("No data available for the selected filters.")
        return
    
    ########################
    # (A) Your Existing Code
    ########################
    grouped = all_pitches.groupby('Batter')
    rows = []

    for batter, group_data in grouped:
        total_events = len(group_data)
        
        # Basic Metrics
        avg_ev = group_data['Exitspeed'].mean() if 'Exitspeed' in group_data else np.nan
        max_ev = group_data['Exitspeed'].max() if 'Exitspeed' in group_data else np.nan
        avg_launch_angle = group_data['Angle'].mean() if 'Angle' in group_data else np.nan
        hard_hit_count = (group_data['Exitspeed'] > 90).sum()
        hard_hit_pct = hard_hit_count / total_events if total_events > 0 else np.nan
        
        barrel_mask = (group_data['Exitspeed'] >= 99) & (group_data['Angle'].between(25, 31))
        barrel_count = barrel_mask.sum()
        barrel_pct = barrel_count / total_events if total_events > 0 else np.nan
        
        gb_count = (group_data['Angle'] < 0).sum()
        gb_pct = gb_count / total_events if total_events > 0 else np.nan
        
        # Pull Percentage
        if 'Batterside' in group_data.columns and not group_data['Batterside'].isnull().all():
            batter_sides = group_data['Batterside'].dropna().unique()
            batter_side = batter_sides[0] if len(batter_sides) > 0 else None
        else:
            batter_side = None

        if batter_side == 'L':
            pull_count = (group_data['Direction'] > 0).sum()
        elif batter_side == 'R':
            pull_count = (group_data['Direction'] < 0).sum()
        else:
            pull_count = np.nan

        pull_pct = pull_count / total_events if total_events > 0 and not np.isnan(pull_count) else np.nan

        pop_fly_count = ((group_data['Angle'] > 50) & (group_data['Exitspeed'] < 85)).sum()
        pop_fly_pct = pop_fly_count / total_events if total_events > 0 else np.nan

        # Calculate avg_xSLG only for rows where Swing is 'Swing'
        avg_xSLG = (
            group_data.loc[group_data['Swing'] == 'Swing', 'xSLG'].mean()
            if 'xSLG' in group_data.columns else np.nan
        )

        # Plate Discipline Metrics
        o_swing = ((group_data['Zone'] == 'Out') & (group_data['Swing'] == 'Swing')).sum()
        z_swing = ((group_data['Zone'] == 'InZone') & (group_data['Swing'] == 'Swing')).sum()
        total_swings = (group_data['Swing'] == 'Swing').sum()
        total_pitches = len(group_data)

        o_contact = ((group_data['Zone'] == 'Out') & (group_data['Swing'] == 'Swing') & (group_data['Contact'] == 'Yes')).sum()
        z_contact = ((group_data['Zone'] == 'InZone') & (group_data['Swing'] == 'Swing') & (group_data['Contact'] == 'Yes')).sum()
        total_contact = (group_data['Contact'] == 'Yes').sum()

        zone_pitches = (group_data['Zone'] == 'InZone').sum()
        first_strike = ((group_data['Pitchcall'] == 'Strike') & (group_data['Swing'] == 'Take')).sum()
        swinging_strike = ((group_data['Swing'] == 'Swing') & (group_data['Contact'] == 'No')).sum()

        # Avoid division by zero with np.maximum
        o_swing_pct = o_swing / np.maximum((group_data['Zone'] == 'Out').sum(), 1)
        z_swing_pct = z_swing / np.maximum((group_data['Zone'] == 'InZone').sum(), 1)
        swing_pct = total_swings / np.maximum(total_pitches, 1)
        o_contact_pct = o_contact / np.maximum(o_swing, 1)
        z_contact_pct = z_contact / np.maximum(z_swing, 1)
        contact_pct = total_contact / np.maximum(total_swings, 1)
        zone_pct = zone_pitches / np.maximum(total_pitches, 1)
        f_strike_pct = first_strike / np.maximum(group_data['Atbatid'].nunique(), 1)
        swstr_pct = swinging_strike / np.maximum(total_pitches, 1)

        # Formatting functions
        def fmt_num(val, decimals=2):
            return f"{val:.{decimals}f}" if pd.notna(val) else np.nan

        def fmt_pct(val):
            return f"{val*100:.2f}%" if pd.notna(val) else np.nan

        def fmt_xslg(val):
            return f"{val:.3f}" if pd.notna(val) else np.nan

        rows.append({
            'Batter': batter,
            'Avg EV': fmt_num(avg_ev),
            'Max EV': fmt_num(max_ev),
            'Avg LA': fmt_num(avg_launch_angle),
            'xSLG': fmt_xslg(avg_xSLG),
            'Hard Hit%': fmt_pct(hard_hit_pct),
            'Barrel%': fmt_pct(barrel_pct),
            'O-Swing%': fmt_pct(o_swing_pct),
            'Z-Swing%': fmt_pct(z_swing_pct),
            'Swing%': fmt_pct(swing_pct),
            'O-Contact%': fmt_pct(o_contact_pct),
            'Z-Contact%': fmt_pct(z_contact_pct),
            'Contact%': fmt_pct(contact_pct),
            'Zone%': fmt_pct(zone_pct),
            'F-Strike%': fmt_pct(f_strike_pct),
            'SwStr%': fmt_pct(swstr_pct),
            'GB%': fmt_pct(gb_pct),
            'PULL%': fmt_pct(pull_pct),
            'POP FLY%': fmt_pct(pop_fly_pct)
        })

    metrics_df = pd.DataFrame(rows)

    ###############################
    # (B) Compute Decision Values
    ###############################
    # 1) We'll define a small helper:
    def apply_20_80_scale(mean_pred, mu, std):
        if pd.isna(mean_pred):
            return np.nan
        if std == 0:
            return 50  # fallback if no variance
        z = (mean_pred - mu) / std
        return np.clip(50 + 10*z, 20, 80)

    # 2) Group by Batter for No-Swing, Swing, and Overall
    #    We assume your code already created 'y_pred_no_swing' (for Take),
    #    'y_pred_swing' (for Swing), and a combined 'y_pred' if desired.

    # No-Swing subset
    df_no_swing_sub = all_pitches[all_pitches['Swing'] == 'Take']
    df_no_swing_group = df_no_swing_sub.groupby('Batter').agg(
        mean_pred_no_swing=('decision_rv_no_swing','mean'),
        pitches_no_swing=('decision_rv_no_swing','count')
    ).reset_index()

    # Swing subset
    df_swing_sub = all_pitches[all_pitches['Swing'] == 'Swing']
    df_swing_group = df_swing_sub.groupby('Batter').agg(
        mean_pred_swing=('decision_rv_swing','mean'),
        pitches_swing=('decision_rv_swing','count')
    ).reset_index()

    # Overall subset
    # If you assigned a combined 'y_pred' to every pitch, we can just average that:
    df_overall_group = all_pitches.groupby('Batter').agg(
        mean_pred_overall=('decision_rv','mean'),
        pitches_overall=('decision_rv','count')
    ).reset_index()

    # 3) Merge them side-by-side
    df_dv_merged = pd.merge(df_no_swing_group, df_swing_group, on='Batter', how='outer')
    df_dv_merged = pd.merge(df_dv_merged, df_overall_group, on='Batter', how='outer')

    # (Optional) Filter out players with <100 total pitches
    #df_dv_merged = df_dv_merged[df_dv_merged['pitches_overall'] >= 100]

    # 4) Reference stats from your big dataset
    #    (You must fill in real values here!)
    mu_no_swing  = 0.0119
    std_no_swing = 0.0199
    mu_swing     = -0.0194
    std_swing    = 0.0129
    mu_overall   = -0.0032
    std_overall  = 0.0130


    # 5) Apply the 20–80 scale
    df_dv_merged['decision_value_no_swing'] = df_dv_merged['mean_pred_no_swing'].apply(
        lambda x: apply_20_80_scale(x, mu_no_swing, std_no_swing)
    )
    df_dv_merged['decision_value_swing'] = df_dv_merged['mean_pred_swing'].apply(
        lambda x: apply_20_80_scale(x, mu_swing, std_swing)
    )
    df_dv_merged['decision_value_overall'] = df_dv_merged['mean_pred_overall'].apply(
        lambda x: apply_20_80_scale(x, mu_overall, std_overall)
    )

    # Round each to 1 decimal place
    df_dv_merged['decision_value_no_swing'] = df_dv_merged['decision_value_no_swing'].round(1)
    df_dv_merged['decision_value_swing']    = df_dv_merged['decision_value_swing'].round(1)
    df_dv_merged['decision_value_overall']  = df_dv_merged['decision_value_overall'].round(1)

    # 6) Merge the decision-value subset into 'metrics_df'
    #    We'll keep it separate to avoid overwriting your existing columns
    final_df = metrics_df.merge(
        df_dv_merged[[
            'Batter',
            'pitches_no_swing','pitches_swing','pitches_overall',
            'decision_value_no_swing','decision_value_swing','decision_value_overall'
        ]],
        on='Batter',
        how='left'
    )

    ###############################
    # (C) Display the combined table
    ###############################
    st.write("### Hitter Metrics + Decision Values (20–80)")
    st.dataframe(final_df.fillna('N/A'))


def calculate_zone_metrics(data):
    """
    Calculate Swing%, Contact%, xSLG (only on swings), and Hard Hit% for each PlateZone.
    """
    if 'PlateZone' not in data.columns:
        st.error("The column 'PlateZone' does not exist in the dataset.")
        return

    # List of valid zones
    zones = ['Heart', 'Shadow', 'Chase', 'Waste']
    zone_metrics = []

    for zone in zones:
        # Filter data for the current zone
        zone_data = data[data['PlateZone'] == zone]
        zone_swings = zone_data[zone_data['Swing'] == 'Swing']
        
        total_pitches = len(zone_data)
        swings = len(zone_swings)
        contacts = (zone_swings['Contact'] == 'Yes').sum()
        hard_hits = (zone_swings['Exitspeed'] > 90).sum() if 'Exitspeed' in zone_swings.columns else 0
        xslg = zone_swings['xSLG'].mean() if not zone_swings['xSLG'].isnull().all() else np.nan
        
        # Calculate Metrics
        swing_pct = swings / total_pitches if total_pitches > 0 else 0
        contact_pct = contacts / swings if swings > 0 else 0
        hard_hit_pct = hard_hits / swings if swings > 0 else 0
        
        zone_metrics.append({
            'Zone': zone,
            'Total Pitches': total_pitches,
            'Swing%': round(swing_pct, 4),
            'Contact%': round(contact_pct, 4),
            'xSLG': round(xslg, 4) if pd.notnull(xslg) else 'N/A',
            'Hard Hit%': round(hard_hit_pct, 4)
        })
    
    # Convert results to DataFrame
    zone_metrics_df = pd.DataFrame(zone_metrics)
    
    # Narrow columns via CSS
    narrow_style = """
    <style>
    table td, table th {
        width: 60px !important;   /* set narrower width */
        text-align: center !important;
    }
    </style>
    """

    st.write("### Zone Metrics Overview")
    st.markdown(narrow_style, unsafe_allow_html=True)
    st.dataframe(zone_metrics_df)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Heatmaps", "Pitch Locations by Playresult" , "Pitch Locations by Decision Value","Spray Chart", "Hitter Metrics",  "Zone Metrics" ,"Raw Data"])

if page == "Heatmaps":
    st.title("Hitter Heatmaps")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Angle Heatmap
    if 'Angle' in filtered_data.columns and not filtered_data['Angle'].isnull().all():
        create_heatmap(filtered_data, 'Angle', axs[0])
    else:
        axs[0].set_title("Launch Angle")
        axs[0].axis('off')
        axs[0].text(0.5, 0.5, "Launch Angle Heatmap\n(Data Not Available)",
                    horizontalalignment='center', verticalalignment='center')

    # Exit Speed Heatmap
    if 'Exitspeed' in filtered_data.columns and not filtered_data['Exitspeed'].isnull().all():
        create_heatmap(filtered_data, 'Exitspeed', axs[1])
    else:
        axs[1].set_title("Exit Velocity")
        axs[1].axis('off')
        axs[1].text(0.5, 0.5, "Exit Velocity Heatmap\n(Data Not Available)",
                    horizontalalignment='center', verticalalignment='center')

    # xSLG Heatmap
    if 'xSLG' in filtered_data.columns and not filtered_data['xSLG'].isnull().all():
        create_heatmap(filtered_data, 'xSLG', axs[2])
    else:
        axs[2].set_title("xSLG")
        axs[2].axis('off')
        axs[2].text(0.5, 0.5, "xSLG Heatmap\n(Data Not Available)",
                    horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    st.pyplot(fig)

elif page == "Spray Chart":
    st.title("Spray Chart")
    spray_data = filtered_data[
        filtered_data['Direction'].notnull() &
        filtered_data['Distance'].notnull()
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    create_spray_chart(spray_data, ax)
    st.pyplot(fig)

elif page == "Pitch Locations by Playresult":
    st.title("Pitch Locations by Playresult")
    plot_pitch_locations_by_playresult(all_pitches)

elif page == "Pitch Locations by Decision Value":
    st.title("Pitch Locations by Decision Value")
    plot_pitch_locations_by_hand_and_ypred(all_pitches)

elif page == "Raw Data":
    st.title("Raw Data Display")

  
    st.write("### Raw Data ")
    if all_pitches.empty:
        st.warning("No filtered data available. Adjust your filters to see results.")
    else:
        st.dataframe(all_pitches.head(1000))  # Display the first 100 rows of the filtered dataset

elif page == "Hitter Metrics":
    st.title("Hitter Metrics")
    display_hitter_metrics(all_pitches)

elif page == "Zone Metrics":
    st.title("Zone Metrics: Heart, Shadow, Chase, Waste")
    calculate_zone_metrics(all_pitches)

