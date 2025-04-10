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
from matplotlib.patches import Polygon, Rectangle
import matplotlib.lines as mlines
import xgboost as xgb
from statsmodels.nonparametric.kernel_regression import KernelReg
try:
    from scipy.stats import gaussian_kde
except ImportError as e:
    print(f"Error importing gaussian_kde: {e}")
import matplotlib as mpl

############################################
# Load Machine Learning Models
############################################
booster = xgb.Booster()
booster.load_model('xSLG_model.json')

best_model = xgb.XGBRegressor()
best_model._Booster = booster  # Assign the booster to the regressor

# Load the No-Swing model (JSON)
no_swing_booster = xgb.Booster()
no_swing_booster.load_model('model_no_swing.json')
model_no_swing = xgb.XGBRegressor()
model_no_swing._Booster = no_swing_booster

# Load the Swing model (JSON)
swing_booster = xgb.Booster()
swing_booster.load_model('model_swing.json')
model_swing = xgb.XGBRegressor()
model_swing._Booster = swing_booster

############################################
# Custom Color Palette and Plot Styling
############################################
kde_min = '#236abe'
kde_mid = '#fefefe'
kde_max = '#a9373b'

kde_palette = (sns.color_palette(f'blend:{kde_min},{kde_mid}', n_colors=1001)[:-1] +
               sns.color_palette(f'blend:{kde_mid},{kde_max}', n_colors=1001)[:-1])

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

############################################
# Load CSV With New Column Names and Rename Accordingly
############################################
# (Update file_path to your new CSV source if needed)
file_path = "https://raw.githubusercontent.com/tdub29/hitterapp/refs/heads/main/USDHITTINGYTD.csv"
df = pd.read_csv(file_path)

# Use a mapping dictionary to rename new columns to the ones the code expects:
# (All keys are assumed case-insensitive; adjust if needed.)
rename_mapping = {
    "gamedate": "Date",
    "batter": "Batter",
    "pitcher": "Pitcher",
    "exitvelocity": "Exitspeed",   # use the provided exit velocity
    "launchang": "Angle",
    "px": "Platelocside",
    "pz": "Platelocheight",
    "pitchoutcome": "Pitchcall",
    "pitchresult": "Playresult",
    "batterhand": "Batterside",
    "pitcherhand": "Pitcherhand",
    "uniqpitchid": "Pitchuid",
    "pitchtypefull": "Autopitchtype",
    "count": "Count",
    "abnumingame": "Paofinning",
    "inn": "Inning",
    "exitdir": "Direction",
    "dist": "Distance"
}

# Convert column names to lower-case (strip spaces) for matching keys
df.columns = [col.strip() for col in df.columns]
# Prepare a dictionary using lower() for matching
current_cols = {col.lower(): col for col in df.columns}
final_mapping = {}
for key, new_name in rename_mapping.items():
    if key in current_cols:
        final_mapping[current_cols[key]] = new_name

df.rename(columns=final_mapping, inplace=True)

############################################
# Create Derived Columns
############################################
# Split "Count" (e.g., "0-0") into numeric Balls and Strikes.
# (Assumes count is in the format "balls-strikes".)
df[['Balls', 'Strikes']] = df['Count'].str.split('-', expand=True).astype(int)
# Now rename these to lower-case as used later:
df.rename(columns={"Balls": "balls", "Strikes": "strikes"}, inplace=True)

# We no longer need to infer pitcher hand from "Relside" because "Pitcherhand" is provided.
# (Similarly, Batterside is provided as "batterHand" renamed to "Batterside".)

############################################
# Pitch Category Grouping
############################################
pitch_categories = {
    "Breaking Ball": ["Slider", "Curveball"],
    "Fastball": ["Fastball", "Four-Seam", "Sinker", "Cutter"],
    "Offspeed": ["ChangeUp", "Splitter"]
}

def categorize_pitch_type(pitch_type):
    for category, pitches in pitch_categories.items():
        if pitch_type in pitches:
            return category
    return "Other"

df['Pitchcategory'] = df['Autopitchtype'].apply(categorize_pitch_type)

############################################
# Create Boolean Count Category Columns
############################################
df['Firstpitch'] = (df['balls'] == 0) & (df['strikes'] == 0)
df['Twostrike'] = df['strikes'] == 2
df['Threeball'] = df['balls'] == 3
df['Evencount'] = (df['balls'] == df['strikes']) & (df['balls'] != 0)
df['Hitterfriendly'] = df['balls'] > df['strikes']
df['Pitcherfriendly'] = df['strikes'] > df['balls']

############################################
# Ensure Numeric Columns Are Correctly Typed
############################################
df['Exitspeed'] = pd.to_numeric(df['Exitspeed'], errors='coerce')
df['Angle'] = pd.to_numeric(df['Angle'], errors='coerce')

############################################
# Convert Date Column and Filter by Date
############################################
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
min_date = df['Date'].min()
max_date = df['Date'].max()
start_date, end_date = st.sidebar.date_input("Select Date Range", value=[min_date, max_date])
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]

############################################
# Streamlit Sidebar Filters
############################################
st.sidebar.header("Filter Options")

batters = df["Batter"].dropna().unique().tolist()
batters.sort()
batters = ["All Hitters"] + batters
selected_batters = st.sidebar.multiselect("Select Batter(s)", batters, default=["All Hitters"])

pitcher_hands = ["All", "R", "L"]
selected_pitcher_hand = st.sidebar.selectbox("Pitcher Hand", pitcher_hands, index=0)

# Higher-level Pitch Categories Filter
pitch_categories_list = list(pitch_categories.keys())
if 'Other' not in pitch_categories_list:
    pitch_categories_list.append('Other')
selected_categories = st.sidebar.multiselect("Select Pitch Category(s)", pitch_categories_list, default=pitch_categories_list)

# Specific Pitch Types Filter – populate from the selected categories.
available_pitch_types = []
for category in selected_categories:
    if category in pitch_categories:
        available_pitch_types.extend(pitch_categories[category])
    else:
        # For 'Other', select pitch types not in any known category.
        categorized = [p for pitches in pitch_categories.values() for p in pitches]
        other_types = df[~df['Autopitchtype'].isin(categorized)]['Autopitchtype'].dropna()
        available_pitch_types.extend(other_types.astype(str).str.strip().tolist())

available_pitch_types = sorted(list(set([pt for pt in available_pitch_types if pt and pt.lower() != 'nan'])))
selected_pitch_types = st.sidebar.multiselect("Select Pitch Type(s)", available_pitch_types, default=available_pitch_types)

# Map count options for later use (if needed)
count_option_to_column = {
    '1st-pitch': 'Firstpitch',
    '2-Strike': 'Twostrike',
    '3-Ball': 'Threeball',
    'Even': 'Evencount',
    'Hitter-Friendly': 'Hitterfriendly',
    'Pitcher-Friendly': 'Pitcherfriendly'
}

# Create Event Column (combining Pitchcall and Playresult)
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

############################################
# Define Swing vs Take based on Exitspeed/Pitchcall
############################################
df['Swing'] = np.where(
    (df['Exitspeed'] > 0) | (df['Pitchcall'].str.contains('Swing|Foul', case=False, na=False)),
    'Swing',    # Swing if there is a nonzero Exitspeed or "Swing"/"Foul" in call.
    'Take'
)

############################################
# Define Strike Zone and Zone Column
############################################
STRIKE_ZONE_SIDE = (-0.83, 0.83)
STRIKE_ZONE_HEIGHT = (1.5, 3.5)
df['Zone'] = np.where(
    (df['Platelocside'].between(*STRIKE_ZONE_SIDE)) &
    (df['Platelocheight'].between(*STRIKE_ZONE_HEIGHT)),
    'InZone',
    'Out'
)

############################################
# Create Unique At-Bat ID Using Date, Pitcher, Paofinning, Inning
############################################
df['Atbatid'] = (
    df['Date'].astype(str).fillna('') + '_' +
    df['Pitcher'].fillna('').astype(str) + '_' +
    df['Paofinning'].fillna('').astype(str) + '_' +
    df['Inning'].fillna('').astype(str)
)

############################################
# Contact and Whiff Columns Based on Exitspeed
############################################
# Whiff = “Yes” if Event string contains “Swinging”
df['Whiff'] = np.where(
    df['Event'].str.contains('Swinging', case=False, na=False),
    'Yes',
    'No'
)

# Contact = “Yes” if the pitch was swung at AND Whiff == “No”
df['Contact'] = np.where(
    (df['Swing'] == 'Swing') & (df['Whiff'] == 'No'),
    'Yes',
    'No'
)


############################################
# Create Final Count String (if needed)
############################################
df['Count'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
df['ContactPct'] = np.where(
    df['Swing'] == 'Swing',
    np.where(df['Contact'] == 'Yes', 1.0, 0.0),
    np.nan
)

############################################
# Map Plate Zones (Heart, Shadow, Chase, Waste) Based on Plate Location
############################################
def map_plate_zone(row):
    side = row['Platelocside']
    height = row['Platelocheight']
    HEART_SIDE = (-0.56, 0.56)
    HEART_HEIGHT = (1.83, 3.17)
    SHADOW_SIDE = (-1.11, 1.11)
    SHADOW_HEIGHT = (1.17, 3.83)
    CHASE_SIDE = (-1.67, 1.67)
    CHASE_HEIGHT = (0.5, 4.33)
    if HEART_SIDE[0] <= side <= HEART_SIDE[1] and HEART_HEIGHT[0] <= height <= HEART_HEIGHT[1]:
        return 'Heart'
    elif SHADOW_SIDE[0] <= side <= SHADOW_SIDE[1] and SHADOW_HEIGHT[0] <= height <= SHADOW_HEIGHT[1]:
        return 'Shadow'
    elif CHASE_SIDE[0] <= side <= CHASE_SIDE[1] and CHASE_HEIGHT[0] <= height <= CHASE_HEIGHT[1]:
        return 'Chase'
    else:
        return 'Waste'

df['PlateZone'] = df.apply(map_plate_zone, axis=1)

############################################
# Split Data into Swing and Take for Model Predictions
############################################
df_no_swing = df[df['Swing'] == 'Take'].dropna(subset=['Platelocside', 'Platelocheight', 'strikes', 'balls']).copy()
df_swing = df[df['Swing'] == 'Swing'].dropna(subset=['Platelocside', 'Platelocheight', 'strikes', 'balls']).copy()

df_no_swing['decision_rv'] = model_no_swing.predict(df_no_swing[['Platelocside','Platelocheight','strikes','balls']])
df_swing['decision_rv'] = model_swing.predict(df_swing[['Platelocside','Platelocheight','strikes','balls']])

# Merge predictions back if "Pitchuid" exists
if 'Pitchuid' in df.columns:
    df_no_swing_merge = df_no_swing[['Pitchuid','decision_rv']].rename(columns={'decision_rv': 'decision_rv_no_swing'})
    df_swing_merge = df_swing[['Pitchuid','decision_rv']].rename(columns={'decision_rv': 'decision_rv_swing'})
    df = df.merge(df_no_swing_merge, on='Pitchuid', how='left').merge(df_swing_merge, on='Pitchuid', how='left')
    df['decision_rv'] = df['decision_rv_no_swing'].combine_first(df['decision_rv_swing'])
else:
    pass

############################################
# 3) Construct the actual boolean filter for Batter
############################################
if selected_batters == ["All Hitters"] or not selected_batters:
    batter_filter = True
else:
    batter_filter = df["Batter"].isin(selected_batters)

############################################
# 4) Construct the pitcher hand filter
############################################
if selected_pitcher_hand == "All":
    pitcher_hand_filter = True
else:
    pitcher_hand_filter = df["Pitcherhand"] == selected_pitcher_hand

############################################
# Apply Overall Filters to Create Final Data Subset
############################################
all_pitches = df[
    (batter_filter) &
    (pitcher_hand_filter) &
    df['Pitchcategory'].isin(selected_categories) &
    df['Autopitchtype'].isin(selected_pitch_types)
]

filtered_data = all_pitches[(all_pitches['Exitspeed'] > 0) & (all_pitches['Exitspeed'].notnull())]
if selected_pitcher_hand != 'All':
    filtered_data = filtered_data[filtered_data['Pitcherhand'] == selected_pitcher_hand]

############################################
# xSLG Prediction: Create Interaction & Predict
############################################
filtered_data['interaction'] = filtered_data['Exitspeed'] * filtered_data['Angle']
X_pred = filtered_data[['Exitspeed', 'Angle', 'interaction']].copy()
X_pred.rename(columns={'Exitspeed':'launch_speed', 'Angle':'launch_angle'}, inplace=True)
filtered_data['xSLG'] = best_model.predict(X_pred)
if 'Pitchuid' in filtered_data.columns and 'Pitchuid' in all_pitches.columns:
    xSLG_data = filtered_data[['Pitchuid', 'xSLG']].drop_duplicates(subset='Pitchuid')
    all_pitches = all_pitches.merge(xSLG_data, on='Pitchuid', how='left')
    all_pitches['xSLG'] = all_pitches['xSLG'].fillna(np.nan)


############################################
# Visualization Functions
############################################
def create_heatmap(data, metric, ax):
    if data.empty or metric not in data.columns:
        ax.set_title(f"No data available for {metric}.")
        ax.axis('off')
        return
    x_min, x_max = -2.5, 2.5
    y_min, y_max = 0, 5
    x_bins = np.linspace(x_min, x_max, 10)
    y_bins = np.linspace(y_min, y_max, 10)
    heatmap_data, _, _ = np.histogram2d(
        data['Platelocside'], data['Platelocheight'],
        bins=[x_bins, y_bins],
        weights=data[metric],
        density=False
    )
    counts, _, _ = np.histogram2d(
        data['Platelocside'], data['Platelocheight'],
        bins=[x_bins, y_bins]
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data = np.divide(heatmap_data, counts, out=np.full_like(heatmap_data, np.nan), where=counts != 0)
    heatmap_data = np.ma.masked_invalid(heatmap_data)
    if metric == 'xSLG' and np.isnan(heatmap_data).all():
        ax.set_title("No data available for xSLG.")
        ax.axis('off')
        return
    if metric == 'Exitspeed':
        vmin, vmax = 60, 100
    elif metric == 'Angle':
        vmin, vmax = -20, 40
    elif metric == 'xSLG':
        vmin, vmax = 0.25, 0.65
    elif metric == 'decision_rv':
        vmin, vmax = -0.2, 0.15
    elif metric == 'ContactPct':
        vmin, vmax = 0.7, 1.0
    else:
        vmin, vmax = np.nanmin(heatmap_data), np.nanmax(heatmap_data)
    extent = [x_min, x_max, y_min, y_max]
    im = ax.imshow(heatmap_data.T, cmap='coolwarm', origin='lower', extent=extent, aspect='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)
    ax.add_patch(plt.Rectangle((-0.83, 1.5), 1.66, 2.1, edgecolor='black', facecolor='none', lw=2))
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
    data['Swing'] = data['Swing'].astype('category')
    data['xSLG'] = pd.to_numeric(data['xSLG'].fillna(0), errors='coerce').fillna(0)
    data['Count'] = data['balls'].astype(str) + '-' + data['strikes'].astype(str)
    pitcher_sides = ['R', 'L']
    swing_types = ['Swing', 'Take']
    plate_vertices = [(-0.83, 0.1), (0.83, 0.1), (0.65, 0.25), (0, 0.5), (-0.65, 0.25)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 16), sharey=True, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    axes = axes.flatten()
    norm = plt.Normalize(vmin=0, vmax=0.8)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    zone_definitions = {
        'Heart': {'x_range': (-0.56, 0.56), 'y_range': (1.83, 3.17), 'color': '#7CFC00', 'alpha': 0.3},
        'Shadow': {'x_range': (-1.11, 1.11), 'y_range': (1.17, 3.83), 'color': '#FFD700', 'alpha': 0.2},
        'Chase': {'x_range': (-1.67, 1.67), 'y_range': (0.5, 4.33), 'color': '#FFA07A', 'alpha': 0.15},
        'Waste': {'x_range': (-2.5, 2.5), 'y_range': (0.0, 5.0), 'color': '#FF6347', 'alpha': 0.1}
    }
    for i, (swing, pitcher_side) in enumerate([(s, p) for s in swing_types for p in pitcher_sides]):
        side_data = data[(data['Swing'] == swing) & (data['Pitcherhand'] == pitcher_side)]
        for _, row in side_data.iterrows():
            marker = 'o' if row['Whiff'] == 'No' else 'x'
            axes[i].scatter(row['Platelocside'], row['Platelocheight'], c=row['xSLG'], cmap='coolwarm',
                            norm=norm, edgecolor='black', s=100, marker=marker, zorder=3)
            if swing == 'Take':
                axes[i].text(row['Platelocside'] - 0.15, row['Platelocheight'], row['Count'],
                             fontsize=8, color='darkblue', ha='right', va='center', zorder=4)
        for zone, props in zone_definitions.items():
            axes[i].add_patch(Rectangle((props['x_range'][0], props['y_range'][0]),
                                          props['x_range'][1] - props['x_range'][0],
                                          props['y_range'][1] - props['y_range'][0],
                                          edgecolor='none', facecolor=props['color'], alpha=props['alpha'], zorder=0))
        axes[i].add_patch(Rectangle((-0.83, 1.5), 1.66, 2.1, edgecolor='black', facecolor='none', lw=2))
        plate = Polygon(plate_vertices, closed=True, linewidth=1, edgecolor='k', facecolor='none')
        axes[i].add_patch(plate)
        axes[i].set_title(f'{swing} vs {pitcher_side}-Handed Pitchers')
        axes[i].set_xlim(-2.5, 2.5)
        axes[i].set_ylim(0, 5)
        axes[i].set_xlabel('PlateLocSide')
        axes[i].set_ylabel('PlateLocHeight')
        if axes[i].get_legend() is not None:
            axes[i].get_legend().remove()
    cbar_ax = fig.add_axes([0.3, 0.5, 0.4, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('xSLG (0 - 0.8)', fontsize=12, color='white')
    cbar.ax.tick_params(labelcolor='white')
    fig.text(cbar_ax.get_position().x0 - 0.15, cbar_ax.get_position().y0, 'X = WHIFF', ha='right', va='center', fontsize=20, color='white')
    plt.subplots_adjust(hspace=0.4)
    st.pyplot(fig)

def plot_pitch_locations_by_hand_and_ypred(data):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    if data.empty:
        st.warning("No data available for the selected filters.")
        return
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
        for event_type in subset['Event'].unique():
            sub_e = subset[subset['Event'] == event_type]
            marker_style = shape_map.get(event_type, 'o')
            sc = axes[i].scatter(sub_e['Platelocside'], sub_e['Platelocheight'],
                                 c=sub_e['decision_rv'], cmap='coolwarm', marker=marker_style,
                                 edgecolor='black', vmin=-0.2, vmax=0.15, s=80, alpha=0.7)
        for zone, props in zone_definitions.items():
            axes[i].add_patch(Rectangle((props['x_range'][0], props['y_range'][0]),
                                          props['x_range'][1]-props['x_range'][0],
                                          props['y_range'][1]-props['y_range'][0],
                                          edgecolor='none', facecolor=props['color'],
                                          alpha=props['alpha'], zorder=0))
        axes[i].add_patch(Rectangle((-0.83,1.5),1.66,2.0, edgecolor='black', facecolor='none', lw=2))
        axes[i].set_title(f"Pitcher Side: {side}")
        axes[i].set_xlim(-2.5,2.5)
        axes[i].set_ylim(0,5)
        axes[i].set_xlabel("PlateLocSide")
        axes[i].set_ylabel("PlateLocHeight")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist())
    cbar.set_label("decision_rv", fontsize=12)
    handles = []
    for event_name, marker_shape in shape_map.items():
        handle = mlines.Line2D([], [], color='darkblue', marker=marker_shape,
                               markersize=8, label=event_name, linestyle='None')
        handles.append(handle)
    legend_obj = fig.legend(handles=handles, loc='right', frameon=True, title='Shapes')
    legend_obj.get_title().set_color('darkblue')
    for text_obj in legend_obj.get_texts():
        text_obj.set_color('darkblue')
    st.pyplot(fig)

def plot_kde_comparison(data):
    f_league_path = "league_kde_earliest.npy"
    x_grid_path = "grid_x.npy"
    y_grid_path = "grid_y.npy"
    try:
        f_league = np.load(f_league_path, allow_pickle=True)
        X = np.load(x_grid_path, allow_pickle=True)
        Y = np.load(y_grid_path, allow_pickle=True)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return
    except Exception as e:
        st.error(f"Error loading .npy files: {e}")
        return
    if data.empty or 'Direction' not in data.columns or 'Angle' not in data.columns:
        st.error("The dataset is missing required columns ('Direction', 'Angle').")
        return
    try:
        x_loc_player = data['Direction']
        y_loc_player = data['Angle']
        values_player = np.vstack([x_loc_player, y_loc_player])
        kernel_player = gaussian_kde(values_player)
        f_player = np.reshape(kernel_player(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)
        f_player = f_player * (100 / f_player.sum())
    except Exception as e:
        st.error(f"Error in KDE computation: {e}")
        return
    try:
        kde_difference = f_player - f_league
        fig, ax = plt.subplots(figsize=(7,7))
        levels = list(range(-13,14,2))
        cfset = ax.contourf(X, Y, kde_difference * 1000, levels=levels, cmap='vlag', extend='both')
        ax.set_xlim(0,90)
        ax.set_ylim(-30,60)
        ax.set_xticks([])
        ax.set_yticks([])
        x_ticks = [0,30,60,90]
        x_labels = ['Pull','Center','Oppo']
        for label, pos0, pos1 in zip(x_labels, x_ticks[:-1], x_ticks[1:]):
            ax.text((pos0+pos1)/2, -35, label, ha='center', va='top', fontsize=12)
        y_ticks = [-30,10,25,50,60]
        y_labels = ['Ground Ball','Line Drive','Fly Ball','Pop Up']
        for label, pos0, pos1 in zip(y_labels, y_ticks[:-1], y_ticks[1:]):
            ax.text(-10, (pos0+pos1)/2, label, ha='right', va='center', fontsize=12)
        sm = plt.cm.ScalarMappable(cmap='vlag', norm=mpl.colors.BoundaryNorm(levels, ncolors=256))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
        cbar.ax.axis('off')
        for label, position, color in zip(['Less\nOften','Same','More\nOften'], [-24,15,53.5],
                                           [sns.color_palette('vlag',25)[0],'k',sns.color_palette('vlag',25)[-1]]):
            cbar.ax.text(1.15, position, label, ha='center', va='center', color=color, fontsize=12, transform=ax.get_yaxis_transform())
        ax.set_title("Batted Ball Difference", fontsize=16)
        fig.text(0.5, 0.01, "batted-ball-charts.streamlit.app | Data: Baseball Savant/pybaseball", ha='center', fontsize=8)
        sns.despine()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in plotting: {e}")

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
    outline_cartesian = [(-distance * np.sin(np.radians(angle)), distance * np.cos(np.radians(angle)))
                         for angle, distance in outline_points]
    outline_x, outline_y = zip(*outline_cartesian)
    ax.plot(outline_x, outline_y, color='red', linewidth=2, label="Field Outline")
    if 'Direction' in data.columns and 'Distance' in data.columns:
        data['Direction_rad'] = np.radians(data['Direction'])
        data['Rotated_X'] = -data['Distance'] * np.sin(data['Direction_rad'])
        data['Rotated_Y'] = data['Distance'] * np.cos(data['Direction_rad'])
        scatter = ax.scatter(data['Rotated_X'], data['Rotated_Y'], c=data['Exitspeed'], cmap='coolwarm', s=50, 
                             edgecolor='k', label='Hits', vmin=60, vmax=100)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Exit Speed")
    ax.set_title("Spray Chart")
    ax.set_xlabel("X (Feet)")
    ax.set_ylabel("Y (Feet)")
    ax.set_xlim([-250, 250])
    ax.set_ylim([0,450])
    ax.set_aspect('equal')
    ax.legend()

def display_hitter_metrics(all_pitches):
    if all_pitches.empty:
        st.write("No data available for the selected filters.")
        return
    grouped = all_pitches.groupby('Batter')
    rows = []
    for batter, group_data in grouped:
        total_events = len(group_data)
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
        if 'Batterside' in group_data.columns and not group_data['Batterside'].isnull().all():
            batter_sides = group_data['Batterside'].dropna().unique()
            batter_side = batter_sides[0] if len(batter_sides) > 0 else None
        else:
            batter_side = None
        if batter_side == 'Left':
            pull_count = (group_data['Direction'] > 0).sum()
        elif batter_side == 'Right':
            pull_count = (group_data['Direction'] < 0).sum()
        else:
            pull_count = np.nan
        pull_pct = pull_count / total_events if total_events > 0 and not np.isnan(pull_count) else np.nan
        pop_fly_count = ((group_data['Angle'] > 50) & (group_data['Exitspeed'] < 85)).sum()
        pop_fly_pct = pop_fly_count / total_events if total_events > 0 else np.nan
        avg_xSLG = group_data.loc[group_data['Swing'] == 'Swing', 'xSLG'].mean() if 'xSLG' in group_data else np.nan
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
        o_swing_pct = o_swing / np.maximum((group_data['Zone'] == 'Out').sum(), 1)
        z_swing_pct = z_swing / np.maximum((group_data['Zone'] == 'InZone').sum(), 1)
        swing_pct = total_swings / np.maximum(total_pitches, 1)
        o_contact_pct = o_contact / np.maximum(o_swing, 1)
        z_contact_pct = z_contact / np.maximum(z_swing, 1)
        contact_pct = total_contact / np.maximum(total_swings, 1)
        zone_pct = zone_pitches / np.maximum(total_pitches, 1)
        f_strike_pct = first_strike / np.maximum(group_data['Atbatid'].nunique(), 1)
        swstr_pct = swinging_strike / np.maximum(total_pitches, 1)
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
    def apply_20_80_scale(mean_pred, mu, std):
        if pd.isna(mean_pred):
            return np.nan
        if std == 0:
            return 50
        z = (mean_pred - mu) / std
        return np.clip(50 + 10*z, 20, 80)
    df_no_swing_sub = all_pitches[all_pitches['Swing'] == 'Take']
    df_no_swing_group = df_no_swing_sub.groupby('Batter').agg(mean_pred_no_swing=('decision_rv_no_swing','mean'),
                                                              pitches_no_swing=('decision_rv_no_swing','count')).reset_index()
    df_swing_sub = all_pitches[all_pitches['Swing'] == 'Swing']
    df_swing_group = df_swing_sub.groupby('Batter').agg(mean_pred_swing=('decision_rv_swing','mean'),
                                                        pitches_swing=('decision_rv_swing','count')).reset_index()
    df_overall_group = all_pitches.groupby('Batter').agg(mean_pred_overall=('decision_rv','mean'),
                                                         pitches_overall=('decision_rv','count')).reset_index()
    df_dv_merged = pd.merge(df_no_swing_group, df_swing_group, on='Batter', how='outer')
    df_dv_merged = pd.merge(df_dv_merged, df_overall_group, on='Batter', how='outer')
    mu_no_swing  = 0.0119
    std_no_swing = 0.0199
    mu_swing     = -0.0194
    std_swing    = 0.0129
    mu_overall   = -0.0032
    std_overall  = 0.0130
    df_dv_merged['decision_value_no_swing'] = df_dv_merged['mean_pred_no_swing'].apply(lambda x: apply_20_80_scale(x, mu_no_swing, std_no_swing))
    df_dv_merged['decision_value_swing'] = df_dv_merged['mean_pred_swing'].apply(lambda x: apply_20_80_scale(x, mu_swing, std_swing))
    df_dv_merged['decision_value_overall'] = df_dv_merged['mean_pred_overall'].apply(lambda x: apply_20_80_scale(x, mu_overall, std_overall))
    df_dv_merged['decision_value_take'] = df_dv_merged['decision_value_no_swing'].round(1)
    df_dv_merged['decision_value_swing'] = df_dv_merged['decision_value_swing'].round(1)
    df_dv_merged['decision_value_overall'] = df_dv_merged['decision_value_overall'].round(1)
    final_df = metrics_df.merge(df_dv_merged[['Batter','pitches_no_swing','pitches_swing','pitches_overall',
                                              'decision_value_take','decision_value_swing','decision_value_overall']],
                                on='Batter', how='right')
    st.write("### Hitter Metrics + Decision Values (20–80)")
    st.dataframe(final_df.fillna('N/A'))

def calculate_zone_metrics(data):
    import numpy as np
    
    # Same function used in display_hitter_metrics:
    def apply_20_80_scale(mean_pred, mu, std):
        if pd.isna(mean_pred):
            return np.nan
        if std == 0:
            return 50
        z = (mean_pred - mu) / std
        return np.clip(50 + 10*z, 20, 80)

    if 'PlateZone' not in data.columns:
        st.error("The column 'PlateZone' does not exist in the dataset.")
        return

    zones = ['Heart', 'Shadow', 'Chase', 'Waste']
    zone_metrics = []
    
    # These are the same 'overall' reference stats used for decision_rv in your example:
    mu_overall = -0.0032
    std_overall = 0.0130

    for zone in zones:
        zone_data = data[data['PlateZone'] == zone]
        zone_swings = zone_data[zone_data['Swing'] == 'Swing']

        total_pitches = len(zone_data)
        swings = len(zone_swings)
        contacts = (zone_swings['Contact'] == 'Yes').sum()

        hard_hits = (zone_swings['Exitspeed'] > 90).sum() if 'Exitspeed' in zone_swings.columns else 0
        xslg = (zone_swings['xSLG'].mean() 
                if 'xSLG' in zone_swings.columns and not zone_swings['xSLG'].isnull().all() 
                else float('nan'))

        # 1) Compute average decision_rv for all pitches in this zone
        dec_rv = (zone_data['decision_rv'].mean() 
                  if 'decision_rv' in zone_data.columns and not zone_data['decision_rv'].isnull().all() 
                  else float('nan'))
        
        # 2) Convert that average decision_rv to 20–80 scale:
        dec_rv_20_80 = apply_20_80_scale(dec_rv, mu_overall, std_overall) if pd.notna(dec_rv) else np.nan

        swing_pct = swings / total_pitches if total_pitches > 0 else 0
        contact_pct = contacts / swings if swings > 0 else 0
        hard_hit_pct = hard_hits / swings if swings > 0 else 0

        zone_metrics.append({
            'Zone': zone,
            'Total Pitches': total_pitches,
            'Swing%': round(swing_pct, 4),
            'Contact%': round(contact_pct, 4),
            'xSLG': round(xslg, 4) if pd.notnull(xslg) else 'N/A',
            'Hard Hit%': round(hard_hit_pct, 4),
            'Decision RV (20–80)': round(dec_rv_20_80, 1) if pd.notnull(dec_rv_20_80) else 'N/A'
        })

    zone_metrics_df = pd.DataFrame(zone_metrics)

    narrow_style = """
    <style>
    table td, table th {
        width: 60px !important;
        text-align: center !important;
    }
    </style>
    """
    st.write("### Zone Metrics Overview")
    st.markdown(narrow_style, unsafe_allow_html=True)
    st.dataframe(zone_metrics_df)

############################################
# Streamlit Navigation
############################################
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Heatmaps", "Pitch Locations by Playresult", "Pitch Locations by Decision Value", "Batted Ball Outcomes", "Spray Chart", "Hitter Metrics", "Zone Metrics", "Raw Data"])

if page == "Heatmaps":
    st.title("Hitter Heatmaps")
    fig, axs = plt.subplots(3, 2, figsize=(18, 28))
    if 'Angle' in filtered_data.columns and not filtered_data['Angle'].isnull().all():
        create_heatmap(filtered_data, 'Angle', axs[0,0])
    else:
        axs[0,0].set_title("Launch Angle")
        axs[0,0].axis('off')
        axs[0,0].text(0.5,0.5,"Launch Angle Heatmap\n(Data Not Available)", horizontalalignment='center', verticalalignment='center')
    if 'Exitspeed' in filtered_data.columns and not filtered_data['Exitspeed'].isnull().all():
        create_heatmap(filtered_data, 'Exitspeed', axs[0,1])
    else:
        axs[0,1].set_title("Exit Velocity")
        axs[0,1].axis('off')
        axs[0,1].text(0.5,0.5,"Exit Velocity Heatmap\n(Data Not Available)", horizontalalignment='center', verticalalignment='center')
    if 'xSLG' in filtered_data.columns and not filtered_data['xSLG'].isnull().all():
        create_heatmap(filtered_data, 'xSLG', axs[1,0])
    else:
        axs[1,0].set_title("xSLG")
        axs[1,0].axis('off')
        axs[1,0].text(0.5,0.5,"xSLG Heatmap\n(Data Not Available)", horizontalalignment='center', verticalalignment='center')
    if 'decision_rv' in all_pitches.columns and not all_pitches['decision_rv'].isnull().all():
        create_heatmap(all_pitches, 'decision_rv', axs[1,1])
    else:
        axs[1,1].set_title("Decision Value")
        axs[1,1].axis('off')
        axs[1,1].text(0.5,0.5,"Decision Value Heatmap\n(Data Not Available)", horizontalalignment='center', verticalalignment='center')
    swing_data = all_pitches[(all_pitches['Swing'] == 'Swing') & (all_pitches['Platelocside'].notnull()) & (all_pitches['Platelocheight'].notnull())]
    if 'ContactPct' in swing_data.columns and not swing_data['ContactPct'].isnull().all():
        create_heatmap(swing_data, 'ContactPct', axs[2,0])
    else:
        axs[2,0].set_title("Contact%")
        axs[2,0].axis('off')
        axs[2,0].text(0.5,0.5,"Contact% Heatmap\n(Data Not Available)", ha='center', va='center')
    plt.tight_layout()
    st.pyplot(fig)
elif page == "Spray Chart":
    st.title("Spray Chart")
    spray_data = filtered_data[(filtered_data['Direction'].notnull()) & (filtered_data['Distance'].notnull())]
    fig, ax = plt.subplots(figsize=(10,8))
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
    if all_pitches.empty:
        st.warning("No filtered data available. Adjust your filters to see results.")
    else:
        st.dataframe(all_pitches.head(1000))
elif page == "Hitter Metrics":
    st.title("Hitter Metrics")
    display_hitter_metrics(all_pitches)
elif page == "Batted Ball Outcomes":
    st.title("Batted Ball Outcomes")
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
    else:
        plot_kde_comparison(filtered_data)
elif page == "Zone Metrics":
    st.title("Zone Metrics: Heart, Shadow, Chase, Waste")
    calculate_zone_metrics(all_pitches)
