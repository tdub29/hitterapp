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
df = df[df['Exitspeed'].notnull()]
df = df[df['Angle'].notnull()]

# Create a mask where 'Exitspeed' and 'Angle' are not NaN
mask = (df['Exitspeed'].notnull()) & (df['Angle'].notnull())

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

df['Event'] = df.apply(lambda row: row['Playresult'] if row['Pitchcall'] == 'InPlay' else row['Pitchcall'], axis=1)

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

# Apply filters using batter_filter
filtered_data = df[
    (batter_filter) &
    (pitcher_filter) &
    (df['Pitchcategory'].isin(selected_categories)) &
    (df['Autopitchtype'].isin(selected_pitch_types)) &
    (df['Exitspeed'] > 0) &
    (df['Exitspeed'].notnull())
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    pitcher_sides = ['R', 'L']
    plate_vertices = [(-0.83, 0.1), (0.83, 0.1), (0.65, 0.25), (0, 0.5), (-0.65, 0.25)]
    
    for i, pitcher_side in enumerate(pitcher_sides):
        side_data = data[data['Pitcherhand'] == pitcher_side]
        sns.scatterplot(
            data=side_data,
            x='Platelocside',
            y='Platelocheight',
            hue='Event',
            palette='coolwarm',
            s=100,
            edgecolor='black',
            ax=axes[i]
        )
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
        
        axes[i].set_title(f'Pitch Locations vs {pitcher_side}-Handed Pitchers')
        axes[i].set_xlim(-2.5, 2.5)
        axes[i].set_ylim(0, 5)
        axes[i].set_xlabel('PlateLocSide')
        axes[i].set_ylabel('PlateLocHeight')
    
    plt.tight_layout()
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

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Heatmaps", "Pitch Locations by Playresult" ,"Spray Chart", "Hitter Metrics"])

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
    plot_pitch_locations_by_playresult(filtered_data)


elif page == "Hitter Metrics":
    st.title("Hitter Metrics")

    if not filtered_data.empty:
        # Group by Batter to display a row per batter
        grouped = filtered_data.groupby('Batter')

        rows = []
        for batter, group_data in grouped:
            total_events = len(group_data)

            avg_ev = group_data['Exitspeed'].mean() if 'Exitspeed' in group_data else np.nan
            max_ev = group_data['Exitspeed'].max() if 'Exitspeed' in group_data else np.nan
            avg_launch_angle = group_data['Angle'].mean() if 'Angle' in group_data else np.nan
            hard_hit_count = (group_data['Exitspeed'] > 90).sum()
            hard_hit_pct = hard_hit_count / total_events if total_events > 0 else np.nan

            barrel_mask = (group_data['Exitspeed'] >= 99) & (group_data['Angle'].between(25,31))
            barrel_count = barrel_mask.sum()
            barrel_pct = barrel_count / total_events if total_events > 0 else np.nan

            ev_90th = 'NA'
            zcontact = 'NA'
            swstrk_pct = 'NA'
            kbb_pct = 'NA'
            contact_pct = 'NA'
            z_swing_chase_pct = 'NA'
            xwoba = 'NA'

            gb_count = (group_data['Angle'] < 0).sum()
            gb_pct = gb_count / total_events if total_events > 0 else np.nan

            if 'Batterside' in group_data.columns and not group_data['Batterside'].isnull().all():
                batter_sides = group_data['Batterside'].dropna().unique()
                if len(batter_sides) > 0:
                    batter_side = batter_sides[0]
                else:
                    batter_side = None
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

            # Format the numeric values:
            def fmt_num(val, decimals=2):
                if pd.isna(val):
                    return np.nan
                return f"{val:.{decimals}f}"

            def fmt_pct(val):
                if pd.isna(val):
                    return np.nan
                return f"{val*100:.2f}%"

            def fmt_xslg(val):
                if pd.isna(val):
                    return np.nan
                return f"{val:.3f}"

            rows.append({
                'Batter': batter,
                'Avg EV': fmt_num(avg_ev),
                'Max EV': fmt_num(max_ev),
                'Avg LA': fmt_num(avg_launch_angle),
                'xSLG': fmt_xslg(avg_xSLG),
                'Hard Hit%': fmt_pct(hard_hit_pct),
                'Barrel%': fmt_pct(barrel_pct),
                '90TH% EV': ev_90th,
                'zCONTACT': zcontact,
                'SwStrk%': swstrk_pct,
                'K-BB%': kbb_pct,
                'CONTACT%': contact_pct,
                'zSWING-CHASE%': z_swing_chase_pct,
                'xWOBA': xwoba,
                'GB%': fmt_pct(gb_pct),
                'PULL%': fmt_pct(pull_pct),
                'POP FLY%': fmt_pct(pop_fly_pct)
            })

        metrics_df = pd.DataFrame(rows)
        st.table(metrics_df)
    else:
        st.write("No data available for the selected filters.")

