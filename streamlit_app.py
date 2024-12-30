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

booster = xgb.Booster()
booster.load_model('xSLG_model.json')

best_model = xgb.XGBRegressor()
best_model._Booster = booster  # Assign the booster to the regressor


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
    df['Pitchcall'] == 'InPlay',
    np.where(
        df['Playresult'] == 'Error',
        df['Taggedhittype'],  # Use Taggedpitchtype if Playresult is 'Error'
        np.where(
            df['Playresult'] == 'Out',
            'Out: ' + df['Taggedhittype'] ,  # Concatenate Taggedpitchtype with "Out"
            df['Playresult']  # Otherwise, use Playresult
        )
    ),
    df['Pitchcall']  # Default to Pitchcall if not 'InPlay'
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


# Map Plate Zones based on PlatelocSide and PlatelocHeight
def map_plate_zone(row):
    side = row['Platelocside']
    height = row['Platelocheight']

    # Heart Zone
    if -6.7 <= side <= 6.7 and 22 <= height <= 38:
        return 'Heart'
    # Shadow Zone
    elif -13.3 <= side <= 13.3 and 14 <= height <= 46:
        return 'Shadow'
    # Chase Zone
    elif -20 <= side <= 20 and 6 <= height <= 52:
        return 'Chase'
    # Waste Zone
    else:
        return 'Waste'

# Apply the mapping function to the dataframe
df['PlateZone'] = df.apply(map_plate_zone, axis=1)




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
    
    # Define pitcher sides and swing types
    pitcher_sides = ['R', 'L']
    swing_types = ['Swing', 'Take']
    plate_vertices = [(-0.83, 0.1), (0.83, 0.1), (0.65, 0.25), (0, 0.5), (-0.65, 0.25)]

    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharey=True, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
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
        
        # Plot scatter points
        scatter = axes[i].scatter(
            side_data['Platelocside'],
            side_data['Platelocheight'],
            c=side_data['xSLG'],
            cmap='coolwarm',
            norm=norm,
            edgecolor='black',
            s=100
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
    plt.subplots_adjust(hspace=0.4, wspace=0.2)  # More space vertically and horizontally
    
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
    Displays hitter metrics in a tabular format on a Streamlit app.
    
    Args:
        filtered_data (pd.DataFrame): Filtered dataset containing hitter data.
    """
    if all_pitches.empty:
        st.write("No data available for the selected filters.")
        return
    
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

        avg_xSLG = group_data['xSLG'].mean() if 'xSLG' in group_data else np.nan

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
    st.dataframe(metrics_df)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Heatmaps", "Pitch Locations by Playresult" ,"Spray Chart", "Hitter Metrics", "Raw Data"])

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

