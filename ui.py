import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
import os
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


# Cache data loading
@st.cache_data
def load_data(file_name):
    csv_path = os.path.join(os.getcwd(), "Cleaned_Dataset", file_name)
    return pd.read_csv(csv_path, low_memory=False)

# Load main dataset
df = load_data("master_race_results.csv")

# Modified Functions
def driver_constructor_performance(df):
    # Top Drivers by Win Ratio
    st.write("Top Drivers by Win Ratio")
    num_drivers = st.sidebar.number_input("Number of Top Drivers", min_value=1, value=5, step=1)
    # Calculate win ratios for drivers
    wins = df[df['position'] == 1].groupby('driverId').size()
    total_races = df.groupby('driverId').size()
    win_ratio = (wins / total_races).fillna(0)
    driver_names = df[['driverId', 'driverRef']].drop_duplicates().set_index('driverId')
    win_ratio_df = win_ratio.to_frame(name='win_ratio').join(driver_names)
    top_drivers = win_ratio_df.sort_values('win_ratio', ascending=False).head(int(num_drivers))
    driver_table = top_drivers.reset_index()[['driverId', 'driverRef', 'win_ratio']]
    # Display the driver table occupying the full container width
    st.dataframe(driver_table, use_container_width=True)

    # Top Constructors by Wins
    st.write("Top Constructors by Wins")
    num_constructors = st.sidebar.number_input("Number of Top Constructors", min_value=1, value=5, step=1)
    # Calculate wins for constructors
    constructor_wins = df[df['position'] == 1].groupby('constructorId').size()
    constructor_names = df[['constructorId', 'constructorRef']].drop_duplicates().set_index('constructorId')
    constructor_wins_df = constructor_wins.to_frame(name='wins').join(constructor_names)
    top_constructors = constructor_wins_df.sort_values('wins', ascending=False).head(int(num_constructors))
    constructor_table = top_constructors.reset_index()[['constructorId', 'constructorRef', 'wins']]
    # Display the constructor table occupying the full container width
    st.dataframe(constructor_table, use_container_width=True)
    
    # Additional formatted string for Top 5 Constructors by Wins (optional)
    constructor_wins = df[df['position'] == 1].groupby('constructorId').size()
    constructor_names = df[['constructorId', 'constructorRef']].drop_duplicates().set_index('constructorId')
    constructor_wins_df = constructor_wins.to_frame(name='wins').join(constructor_names)
    top_constructors = constructor_wins_df.sort_values('wins', ascending=False).head(5)
    constructors_str = "Top 5 Constructors by Wins:\n" + "\n".join(
        f"{row['constructorRef']} (ID: {constructor_id}) - Wins: {int(row['wins'])}"
        for constructor_id, row in top_constructors.iterrows()
    )
    # Uncomment the following line if you wish to display the string as well
    # st.write(constructors_str)
    
    return [], None
def qualifying_vs_race_performance(df):
        # Enhance theme for a better UI
        sns.set_theme(style="whitegrid")
        outputs = []

        # Create a deep copy and calculate position gain (grid - final position)
        df = df.copy()
        df['position_gain'] = df['grid'] - df['position']

        # Let the user specify how many top drivers to compare
        n_top = st.sidebar.number_input("Number of Top Drivers", min_value=1, value=5, step=1)

        # Build a driver mapping to get the driver name
        driver_mapping = df[['driverId', 'driverRef']].drop_duplicates().set_index('driverId')

        # Compute average position gain per driver
        avg_gain = df.groupby('driverId')['position_gain'].mean()

        # Join with driver names and prepare the table
        gain_df = avg_gain.to_frame().join(driver_mapping)
        gain_df = gain_df.reset_index().rename(columns={
            'driverId': 'Driver ID',
            'driverRef': 'Driver Name',
            'position_gain': 'Avg Position Gain'
        })
        gain_df = gain_df.sort_values('Avg Position Gain', ascending=False)
        top_drivers = gain_df.head(int(n_top))

        outputs.append("Top Drivers Qualifying vs. Race Performance")
        outputs.append(top_drivers)

        # Prepare the figure to compare Qualifying vs. Race performance for the top drivers
        fig, ax = plt.subplots(figsize=(12, 7))
        top_driver_ids = top_drivers['Driver ID'].tolist()
        df_top = df[df['driverId'].isin(top_driver_ids)]

        # Remove existing driverRef column to avoid duplicates and merge to get driver names for coloring in the plot
        if 'driverRef' in df_top.columns:
            df_top = df_top.drop(columns=['driverRef'])
        df_top = df_top.merge(driver_mapping, left_on='driverId', right_index=True, how='left')

        # Let the user choose a driver to highlight
        selected_driver = st.sidebar.selectbox("Select Driver to Highlight", df['driverRef'].unique())
        st.sidebar.write("Driver Name:", selected_driver)
        
        # Plot the entire dataset with low opacity
        # Plot all drivers in a uniform grey color with low opacity
        sns.scatterplot(
            data=df,
            x='grid',
            y='position',
            color='gray',
            ax=ax,
            alpha=0.3,
            edgecolor=None,
            legend=False
        )
        
        # Overlay the selected driver's points in red
        highlight = df[df['driverRef'] == selected_driver]
        sns.scatterplot(
            data=highlight,
            x='grid',
            y='position',
            color='red',
            ax=ax,
            alpha=0.9,
            edgecolor="w",
            legend=False,
            s=200
        )
        
        ax.set_title("Qualifying Grid vs. Final Race Position", fontsize=16, weight="bold")
        ax.set_xlabel("Starting Grid Position", fontsize=12)
        ax.set_ylabel("Finishing Position", fontsize=12)
        ax.invert_yaxis()  # Invert y-axis so that higher finishing positions appear lower
        
        plt.tight_layout()
        return outputs, fig
def analyze_pit_stop_strategies(df):
    outputs = []
    # Load pit stop data
    pit_df = load_data("master_pit_stops.csv")
    pit_counts = pit_df.groupby(['raceId', 'driverId']).size().reset_index(name='pit_stops')
    
    # Merge with main race data and fill missing pit stop counts with 0
    merged_df = df.merge(pit_counts, on=['raceId', 'driverId'], how='left')
    merged_df['pit_stops'] = merged_df['pit_stops'].fillna(0)
    
    # Interactive filtering: select finishing position range
    min_pos = int(merged_df['position'].min())
    max_pos = int(merged_df['position'].max())
    pos_range = st.sidebar.slider("Select Finishing Position Range", min_pos, max_pos, (min_pos, max_pos))
    filtered_df = merged_df[(merged_df['position'] >= pos_range[0]) & (merged_df['position'] <= pos_range[1])]
    
    # Show or hide correlation info
    show_corr = st.sidebar.checkbox("Show Correlation Info", value=True)
    if show_corr:
        st.sidebar.markdown("### Correlation Analysis")
        # Calculate correlation between pit stops and finishing position for filtered data
        correlation = filtered_df['pit_stops'].corr(filtered_df['position'])
        if correlation > 0:
            sign = "positive"
            outcome = (
                "This suggests that more pit stops tend to be associated with a worse finishing position, "
                "indicating that additional pit stops may adversely affect race outcomes."
            )
        else:
            sign = "negative"
            outcome = (
                "This suggests that more pit stops tend to be associated with a better finishing position, "
                "which might imply that strategic pit stops can improve race performance."
            )
        abs_corr = abs(correlation)
        if abs_corr < 0.3:
            strength = "weak"
            extra = (
                "while there's a slight tendency for more pit stops to be associated with finishing positions, "
                "this relationship is not very strong."
            )
        elif abs_corr < 0.7:
            strength = "moderate"
            extra = ""
        else:
            strength = "strong"
            extra = ""
            
        md_text = f"""
        
**Correlation Analysis**

- **Correlation Value:** {correlation:.2f} ({sign} correlation, {strength}{' - ' + extra if extra else ''})
- **Outcome:** {outcome}

"""
        outputs.append(md_text)
    # Select plot type
    plot_type = st.sidebar.selectbox("Choose Plot Type", ["Bar Plot", "Scatter Plot"])
    
    # Set up enhanced visualization using seaborn theme and improved aesthetics
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if plot_type == "Bar Plot":
        # Compute average pit stops per finishing position for filtered data
        avg_pits = filtered_df.groupby('position')['pit_stops'].mean().reset_index()
        bar_plot = sns.barplot(
            x='position', 
            y='pit_stops', 
            data=avg_pits, 
            palette='viridis', 
            ax=ax
        )
        # Add annotations on each bar
        for index, row in avg_pits.iterrows():
            ax.text(
                index, 
                row['pit_stops'] + 0.05,
                f"{row['pit_stops']:.2f}",
                ha="center", 
                va="bottom", 
                fontsize=10, 
                color="black", 
                fontweight="bold"
            )
        ax.set_title("Average Pit Stops by Finishing Position", fontsize=16, weight="bold")
    else:
        # Scatter plot: each race's finishing position vs pit stops with a regression line
        scatter_plot = sns.scatterplot(
            data=filtered_df,
            x='position', 
            y='pit_stops', 
            hue='pit_stops', 
            palette='viridis', 
            ax=ax
        )
        sns.regplot(
            data=filtered_df,
            x='position', 
            y='pit_stops', 
            scatter=False, 
            color='red', 
            ax=ax
        )
        ax.set_title("Pit Stops vs Finishing Position", fontsize=16, weight="bold")
    
    ax.set_xlabel("Finishing Position", fontsize=12)
    ax.set_ylabel("Pit Stops", fontsize=12)
    
    plt.tight_layout()
    return outputs, fig
def head_to_head_analysis(df):
    outputs = []

    # Let the user choose the number of top rivalries to display
    n_rivalries = st.sidebar.number_input("Enter number of top rivalries", min_value=1, value=5, step=1)

    # Compute average race positions per driver per race and pivot to create driver matrix
    avg_positions = df.groupby(['driverId', 'raceId'])['position'].mean().reset_index()
    driver_matrix = avg_positions.pivot(index='raceId', columns='driverId', values='position')

    # Compute correlation matrix to gauge similarity in race finishes (optional additional info)
    corr_matrix = driver_matrix.corr()
    corr_matrix.index.name = 'Driver 1 ID'
    corr_matrix.columns.name = 'Driver 2 ID'

    # Use upper-triangular mask to get each pair once (ignoring self-correlation)
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_pairs = corr_matrix.where(mask).stack().reset_index(name='Correlation')

    # Merge to obtain driver names for both drivers
    driver_names = df[['driverId', 'driverRef']].drop_duplicates()
    corr_pairs = corr_pairs.merge(driver_names, left_on='Driver 1 ID', right_on='driverId', how='left')\
                           .rename(columns={'driverRef': 'Driver 1 Name'})\
                           .drop('driverId', axis=1)
    corr_pairs = corr_pairs.merge(driver_names, left_on='Driver 2 ID', right_on='driverId', how='left')\
                           .rename(columns={'driverRef': 'Driver 2 Name'})\
                           .drop('driverId', axis=1)

    # Function to compute average absolute finish difference in common races between two drivers
    def compute_avg_diff(driver1_id, driver2_id, pivot):
        common = pivot[[driver1_id, driver2_id]].dropna()
        if common.empty:
            return np.nan
        return np.abs(common[driver1_id] - common[driver2_id]).mean()

    # Compute head-to-head competitiveness: lower average finish difference implies more competitive rivalry.
    corr_pairs['Avg Finish Diff'] = corr_pairs.apply(
        lambda row: compute_avg_diff(row['Driver 1 ID'], row['Driver 2 ID'], driver_matrix), axis=1
    )

    # Select the top n rivalries based on lowest average finish difference
    top_pairs = corr_pairs.sort_values('Avg Finish Diff', ascending=True).head(int(n_rivalries))
    outputs.append("Top {} Most Competitive Head-to-Head Driver Rivalries (based on race finish closeness):".format(int(n_rivalries)))
    st.dataframe(top_pairs[['Driver 1 ID', 'Driver 1 Name', 'Driver 2 ID', 'Driver 2 Name', 'Correlation', 'Avg Finish Diff']])
    
    # First plot: Bar chart for the top n head-to-head competitiveness (lower diff is better)
    pair_labels = top_pairs.apply(lambda row: f"{row['Driver 1 ID']} & {row['Driver 2 ID']}", axis=1).tolist()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_pairs['Avg Finish Diff'], y=pair_labels, orient='h', ax=ax1, palette='viridis')
    ax1.set_xlabel("Average Finish Difference")
    ax1.set_title("Top {} Competitive Rivalries".format(int(n_rivalries)))
    plt.tight_layout()

    # Create driver selection options from these top rivalries using driverId and driverRef
    drivers_df = df[['driverId', 'driverRef']].drop_duplicates()
    driver_ids_in_rivalries = set()
    for _, row in top_pairs.iterrows():
        driver_ids_in_rivalries.update([row['Driver 1 ID'], row['Driver 2 ID']])
    drivers_df = drivers_df[drivers_df['driverId'].isin(driver_ids_in_rivalries)]
    drivers_df['option'] = drivers_df.apply(
        lambda row: f"{row['driverId']} - {row['driverRef']}", axis=1
    )
    driver_options = drivers_df['option'].tolist()

    st.sidebar.markdown("## Select Two Drivers for Detailed Head-to-Head Analysis")
    if driver_options:
        driver1_option = st.sidebar.selectbox("Driver 1", driver_options, index=0)
        driver2_option = st.sidebar.selectbox("Driver 2", driver_options, index=1 if len(driver_options) > 1 else 0)
        driver1_id = int(driver1_option.split(" - ")[0])
        driver2_id = int(driver2_option.split(" - ")[0])
    else:
        st.sidebar.warning("No drivers available from top rivalries for selection.")
        return outputs, fig1

    # If the same driver is selected, just show the bar plot.
    if driver1_id == driver2_id:
        outputs.append("Please select two different drivers for head-to-head analysis.")
        return outputs, fig1

    # Filter data for the selected drivers and pivot to get common races
    df_selected = df[df['driverId'].isin([driver1_id, driver2_id])]
    df_pivot = df_selected.pivot_table(index='raceId', columns='driverId', values='position')
    df_pivot = df_pivot.dropna()
    if df_pivot.empty or df_pivot.shape[1] < 2:
        outputs.append("Not enough common races between the two selected drivers for head-to-head analysis.")
        return outputs, fig1

    head2head_corr = df_pivot.corr().iloc[0, 1]
    avg_finish_diff_selected = np.abs(df_pivot.iloc[:, 0] - df_pivot.iloc[:, 1]).mean()
    outputs.append("Head-to-Head Analysis between selected drivers:")
    outputs.append(" - Correlation: {:.2f}".format(head2head_corr))
    outputs.append(" - Average Finish Difference: {:.2f} positions".format(avg_finish_diff_selected))

    # Detailed plot for the selected two drivers
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df_pivot.iloc[:, 0], y=df_pivot.iloc[:, 1], ax=ax2, s=100, color='purple')
    ax2.set_title("Head-to-Head Race Positions")
    ax2.set_xlabel(f"Driver {driver1_id} Position")
    ax2.set_ylabel(f"Driver {driver2_id} Position")
    min_val = min(df_pivot.min().min(), df_pivot.max().min())
    max_val = max(df_pivot.max().max(), df_pivot.min().max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.tight_layout()

    # Combine the two plots in a vertical layout
    fig_combined, (axA, axB) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    # Top subplot: Bar chart for top competitive rivalries
    sns.barplot(x=top_pairs['Avg Finish Diff'], y=pair_labels, orient='h', ax=axA, palette='viridis')
    axA.set_xlabel("Average Finish Difference")
    axA.set_title("Top {} Competitive Rivalries".format(int(n_rivalries)))
    # Bottom subplot: Detailed scatter plot for the two selected drivers
    sns.scatterplot(x=df_pivot.iloc[:, 0], y=df_pivot.iloc[:, 1], ax=axB, s=100, color='purple')
    axB.set_title("Head-to-Head Race Positions")
    axB.set_xlabel(f"Driver {driver1_id} Position")
    axB.set_ylabel(f"Driver {driver2_id} Position")
    axB.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.tight_layout()

    # Display detailed race results as a dataframe
    st.dataframe(df_pivot.reset_index())

    return outputs, fig_combined
def hypothetical_driver_swap(df, driver1, driver2):
    outputs = []
    # Ensure data types are correct
    df['year'] = df['year'].astype(int)
    df['constructorId'] = df['constructorId'].astype(int)
    # Convert position to numeric, handling non-numeric values as NaN
    df['position'] = pd.to_numeric(df['position'], errors='coerce')

    # Get driver IDs (case-insensitive)
    driver1_id_array = df[df['driverRef'].str.lower() == driver1.lower()]['driverId'].unique()
    driver2_id_array = df[df['driverRef'].str.lower() == driver2.lower()]['driverId'].unique()
    if len(driver1_id_array) == 0 or len(driver2_id_array) == 0:
        outputs.append(f"❌ **Error:** One or both drivers (_{driver1}_, _{driver2}_) not found in dataset.")
        return outputs, None

    driver1_id, driver2_id = driver1_id_array[0], driver2_id_array[0]
    driver1_name, driver2_name = driver1, driver2  # Use driverRef for outputs

    driver1_seasons = set(df[df['driverId'] == driver1_id]['year'])
    driver2_seasons = set(df[df['driverId'] == driver2_id]['year'])
    common_seasons = list(driver1_seasons.intersection(driver2_seasons))
    if not common_seasons:
        outputs.append("❌ **Error:** No overlapping seasons found for these drivers.")
        return outputs, None

    mask = df['year'].isin(common_seasons)
    driver1_data = df[mask & (df['driverId'] == driver1_id)].copy()
    driver2_data = df[mask & (df['driverId'] == driver2_id)].copy()

    orig_driver1_avg = driver1_data['position'].mean(skipna=True)
    orig_driver2_avg = driver2_data['position'].mean(skipna=True)

    # Pre-calculate team performance: average finishing position per team per year (lower is better)
    team_perf = df.groupby(['year', 'constructorId'])['position'].mean().reset_index()

    # Monte Carlo simulation parameters
    S = 1000  # number of iterations per race
    sim_driver1_values = []
    sim_driver2_values = []

    # For every common season, simulate new finishing positions using swapped teams
    for year in sorted(common_seasons):
        d1_year = driver1_data[driver1_data['year'] == year]
        d2_year = driver2_data[driver2_data['year'] == year]
        if d1_year.empty or d2_year.empty:
            continue
        # Get each driver's constructor for that season
        d1_const = int(d1_year['constructorId'].iloc[0])
        d2_const = int(d2_year['constructorId'].iloc[0])
        # Get team performance averages for that season
        t1 = team_perf[(team_perf['year'] == year) & (team_perf['constructorId'] == d1_const)]
        t2 = team_perf[(team_perf['year'] == year) & (team_perf['constructorId'] == d2_const)]
        if t1.empty or t2.empty:
            continue
        d1_team_perf = t1.iloc[0]['position']
        d2_team_perf = t2.iloc[0]['position']
        # Define adjustment ratios (avoid division by zero)
        ratio1 = d2_team_perf / d1_team_perf if d1_team_perf != 0 else 1
        ratio2 = d1_team_perf / d2_team_perf if d2_team_perf != 0 else 1
        # Simulate for each race with valid position data
        for pos in d1_year['position'].dropna().values:
            sims = np.random.normal(loc=pos * ratio1, scale=0.1 * pos, size=S)
            sim_driver1_values.extend(sims)
        for pos in d2_year['position'].dropna().values:
            sims = np.random.normal(loc=pos * ratio2, scale=0.1 * pos, size=S)
            sim_driver2_values.extend(sims)

    # Check if simulations were performed
    if not sim_driver1_values:
        outputs.append(f"⚠️ **Warning:** No valid races found for simulation for Driver {driver1_name}")
        sim_driver1_avg = float('nan')
    else:
        sim_driver1_avg = np.mean(sim_driver1_values)

    if not sim_driver2_values:
        outputs.append(f"⚠️ **Warning:** No valid races found for simulation for Driver {driver2_name}")
        sim_driver2_avg = float('nan')
    else:
        sim_driver2_avg = np.mean(sim_driver2_values)

    # Build Markdown formatted output with the results
    outputs.append("## Original Average Positions")
    outputs.append(f"**Driver {driver1_name}:** {orig_driver1_avg:.2f}")
    outputs.append(f"**Driver {driver2_name}:** {orig_driver2_avg:.2f}")
    outputs.append("---")
    outputs.append("## Simulated Average Positions after Swap")
    outputs.append(f"**Driver {driver1_name}:** {sim_driver1_avg:.2f}")
    outputs.append(f"**Driver {driver2_name}:** {sim_driver2_avg:.2f}")
    outputs.append("---")
    outputs.append("_Final simulation completed successfully!_")

    # Create a bar chart for side-by-side comparison
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    indices = [0, 1]  # 0 for driver1, 1 for driver2
    width = 0.35

    orig_avgs = [orig_driver1_avg, orig_driver2_avg]
    sim_avgs = [sim_driver1_avg, sim_driver2_avg]

    ax.bar([i - width/2 for i in indices], orig_avgs, width, label='Original', color='skyblue')
    ax.bar([i + width/2 for i in indices], sim_avgs, width, label='Simulated', color='salmon')

    ax.set_xticks(indices)
    ax.set_xticklabels([f"{driver1_name}", f"{driver2_name}"])
    ax.set_ylabel("Average Position")
    ax.set_title("Driver Average Positions: Original vs Simulated")
    ax.legend()
    plt.tight_layout()

    return outputs, fig
def driver_team_network(df):
    outputs = []
    
    # Use a toggle checkbox named "All" to show the network for all drivers
    show_all = st.sidebar.checkbox("All")
    
    team_names = df[['constructorId', 'constructorRef']].drop_duplicates().set_index('constructorId')['constructorRef'].to_dict()
    
    if show_all:
        # Build network for all drivers' team transitions
        driver_history = df[['driverId', 'constructorId', 'year']].sort_values('year')
        G = nx.DiGraph()
        for driver in driver_history['driverId'].unique():
            history = driver_history[driver_history['driverId'] == driver]
            teams = list(history['constructorId'])
            years = list(history['year'])
            for i in range(len(teams) - 1):
                if teams[i] != teams[i + 1]:
                    G.add_edge(teams[i], teams[i + 1], year=years[i])
        title = "Team Transition Network for All Drivers"
        
        # Create a figure for the network
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#86bf91", node_size=1000, alpha=0.9)
        edge_options = {"arrowstyle": "->", "arrowsize": 20, "edge_color": "#555555", "width": 2, "connectionstyle": "arc3,rad=0.1"}
        nx.draw_networkx_edges(G, pos, ax=ax, **edge_options)
        node_labels = {node: team_names.get(node, str(node)) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=11, font_color="black")
        edge_labels = {(u, v): f"{G[u][v]['year']}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="darkred", font_size=9, label_pos=0.3)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")
        plt.tight_layout()
        
        # Build and store a dataframe of the edges of the network
        edges_df = pd.DataFrame(
            [(u, v, d["year"]) for u, v, d in G.edges(data=True)],
            columns=["From", "To", "Year"]
        )
        outputs.append(edges_df)
        return outputs, fig
    else:
        # Let the user select a driver
        driver_names = df[['driverId', 'driverRef']].drop_duplicates()
        options = sorted(driver_names['driverRef'].unique())
        selected_driver = st.sidebar.selectbox(
            "Select a Driver for Team Network",
            options,
            index=options.index("adamich") if "adamich" in options else 0
        )
        # Get the corresponding driverId
        driver_id = driver_names[driver_names['driverRef'] == selected_driver]['driverId'].iloc[0]
        
        # Filter and sort the team history for the selected driver
        driver_history = df[df['driverId'] == driver_id][['constructorId', 'year']].sort_values('year')
        if driver_history.empty:
            outputs.append(pd.DataFrame({"Message": ["No team history found for the selected driver."]}))
            return outputs, None

        # Build a directed graph from the driver's team transitions
        G = nx.DiGraph()
        teams = list(driver_history['constructorId'])
        years = list(driver_history['year'])
        for i in range(len(teams) - 1):
            if teams[i] != teams[i + 1]:
                G.add_edge(teams[i], teams[i + 1], year=years[i])
        title = f"Team Transition Network for {selected_driver}"
        
        # Configure the layout and visualization parameters with enhanced styling
        fig, ax = plt.subplots(figsize=(7, 7))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#86bf91", node_size=1000, alpha=0.9)
        edge_options = {"arrowstyle": "->", "arrowsize": 20, "edge_color": "#555555", "width": 2, "connectionstyle": "arc3,rad=0.1"}
        nx.draw_networkx_edges(G, pos, ax=ax, **edge_options)
        node_labels = {node: team_names.get(node, str(node)) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=11, font_color="black")
        edge_labels = {(u, v): f"{G[u][v]['year']}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="darkred", font_size=9, label_pos=0.3)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")
        plt.tight_layout()
        
        # Build and store a dataframe of the driver's team transitions
        trans_df = pd.DataFrame(
            [(teams[i], teams[i+1], years[i]) for i in range(len(teams)-1) if teams[i] != teams[i+1]],
            columns=["From", "To", "Year"]
        )
        outputs.append(trans_df)
        return outputs, fig

def team_performance_comparison(df):
    outputs = []
    sns.set_theme(style="whitegrid", font_scale=1.1)
    import matplotlib.pyplot as plt

    # Create a mapping from constructorId to constructorRef (team names)
    team_name_mapping = (
        df[['constructorId', 'constructorRef']]
        .drop_duplicates()
        .set_index('constructorId')['constructorRef']
        .to_dict()
    )

    # User inputs to make the analysis dynamic
    n_top_teams = st.sidebar.number_input(
        "Number of Top Teams for Comparison", min_value=2, value=10, step=1
    )
    n_overall_success = st.sidebar.number_input(
        "Number of Teams for Overall Top Success Rate", min_value=1, value=5, step=1
    )
    show_h2h_overall = st.sidebar.checkbox("Display Head-to-Head Overall Comparison", value=False)
    show_h2h_circuit = st.sidebar.checkbox("Display Head-to-Head Circuit Comparison", value=False)

    # Compute overall podium success (i.e. finishing in the top 3)
    team_success = df[df['position'] <= 3].groupby('constructorId').size() / df.groupby('constructorId').size()
    # Map team IDs to team names
    team_success.index = team_success.index.map(team_name_mapping)

    # Compute podium success by circuit
    circuit_team_success = (
        df[df['position'] <= 3]
        .groupby(['circuitId', 'constructorId'])
        .size() /
        df.groupby(['circuitId', 'constructorId']).size()
    )
    circuit_team_success = circuit_team_success.unstack().fillna(0)
    # Update columns to team names
    circuit_team_success.columns = circuit_team_success.columns.map(team_name_mapping)

    # Helper function to compute head-to-head win ratios for a given grouping
    def compute_h2h(df_subset, group_fields):
        wins = defaultdict(int)
        games = defaultdict(int)
        groups = df_subset.groupby(group_fields)
        for name, group in groups:
            # Determine each team's best finishing position in the group
            best_positions = group.groupby('constructorId')['position'].min()
            # Map team IDs to names
            best_positions.index = best_positions.index.map(team_name_mapping)
            teams = best_positions.index.tolist()
            # Compare each ordered pair (i, j)
            for i in teams:
                for j in teams:
                    if i == j:
                        continue
                    games[(i, j)] += 1
                    if best_positions.loc[i] < best_positions.loc[j]:
                        wins[(i, j)] += 1
        # Calculate win ratios
        ratios = {}
        for key in games:
            ratios[key] = wins[key] / games[key] if games[key] > 0 else None
        return ratios

    # Calculate head-to-head comparisons overall and by circuit
    h2h_overall = compute_h2h(df, 'raceId')
    h2h_circuit = compute_h2h(df, ['circuitId', 'raceId'])

    # Select top teams based on overall podium success (using team names)
    top_teams = team_success.sort_values(ascending=False).head(n_top_teams).index.tolist()

    # Helper function to build a matrix (for heatmaps) from head-to-head ratios
    def build_matrix(ratios, teams):
        mat = pd.DataFrame(index=teams, columns=teams, dtype=float)
        mat.index.name = "Constructor"
        for i in teams:
            for j in teams:
                if i == j:
                    mat.loc[i, j] = np.nan
                else:
                    mat.loc[i, j] = ratios.get((i, j), np.nan)
        return mat

    h2h_overall_mat = build_matrix(h2h_overall, top_teams)
    h2h_circuit_mat = build_matrix(h2h_circuit, top_teams)

    # Determine the number of subplots based on user selection
    n_plots = 1  # always show overall podium success bar chart
    if show_h2h_overall:
        n_plots += 1
    if show_h2h_circuit:
        n_plots += 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # Plot 1: Bar Chart for Overall Podium Success for the top teams (by success rate)
    ax0 = axes[0]
    team_success[top_teams].sort_values(ascending=False).plot(
        kind='bar', ax=ax0, color='skyblue', edgecolor='black'
    )
    ax0.set_title("Top {} Teams by Overall Podium Success Rate".format(n_top_teams), fontsize=16, weight='bold')
    ax0.set_xlabel("Team", fontsize=12)
    ax0.set_ylabel("Podium Success Rate", fontsize=12)
    ax0.tick_params(axis='both', which='major', labelsize=10)

    next_ax_index = 1
    # Plot 2: Heatmap for Head-to-Head Overall Win Ratios
    if show_h2h_overall:
        ax_h2h_overall = axes[next_ax_index]
        sns.heatmap(
            h2h_overall_mat, cmap='coolwarm', annot=True, fmt=".2f", ax=ax_h2h_overall,
            cbar_kws={'label': 'Win Ratio'}, linewidths=0.5, linecolor='gray'
        )
        ax_h2h_overall.set_title("Overall Head-to-Head Win Ratio\n(Row wins over Column)", fontsize=16, weight='bold')
        ax_h2h_overall.set_xlabel("Opponent Team", fontsize=12)
        ax_h2h_overall.set_ylabel("Team", fontsize=12)
        ax_h2h_overall.tick_params(axis='both', which='major', labelsize=10)
        next_ax_index += 1

    # Plot 3: Heatmap for Head-to-Head Circuit-Specific Win Ratios
    if show_h2h_circuit:
        ax_h2h_circuit = axes[next_ax_index]
        sns.heatmap(
            h2h_circuit_mat, cmap='coolwarm', annot=True, fmt=".2f", ax=ax_h2h_circuit,
            cbar_kws={'label': 'Win Ratio'}, linewidths=0.5, linecolor='gray'
        )
        ax_h2h_circuit.set_title("Circuit-Specific Head-to-Head Win Ratio\n(Aggregated over Circuits)", fontsize=16, weight='bold')
        ax_h2h_circuit.set_xlabel("Opponent Team", fontsize=12)
        ax_h2h_circuit.set_ylabel("Team", fontsize=12)
        ax_h2h_circuit.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    # Instead of plain text, build dataframes for the outputs
    overall_success_df = team_success.sort_values(ascending=False).head(n_overall_success).reset_index()
    overall_success_df.columns = ["Constructor", "PodiumSuccessRate"]

    circuit_avg_df = pd.DataFrame(circuit_team_success.mean().sort_values(ascending=False).head(n_overall_success)).reset_index()
    circuit_avg_df.columns = ["Constructor", "AvgCircuitSuccessRate"]

    outputs.append(overall_success_df)
    outputs.append(circuit_avg_df)
    if show_h2h_overall:
        outputs.append(h2h_overall_mat.reset_index().rename(columns={"index": "Constructor"}))
    if show_h2h_circuit:
        outputs.append(h2h_circuit_mat.reset_index().rename(columns={"index": "Constructor"}))
    return outputs, fig
def driver_consistency(df):
        # Calculate performance consistency metrics per driver using both IDs and names
        driver_mapping = df[['driverId', 'driverRef']].drop_duplicates().set_index('driverId')['driverRef']
        
        consistency = df.groupby('driverId').agg({'position': ['std', 'mean', 'count']}).reset_index()
        consistency.columns = ['driverId', 'position_std', 'avg_position', 'races']
        min_races = 10
        consistency = consistency[consistency['races'] >= min_races]
        
        # Add driver names into the dataframe
        consistency['driver_name'] = consistency['driverId'].map(driver_mapping)
        
        # Calculate the percentage of top-10 finishes for each driver
        top_10_finishes = df[df['position'] <= 10].groupby('driverId').size()
        total_races = df.groupby('driverId').size()
        top_10_ratio = (top_10_finishes / total_races * 100).round(2)
        consistency['top_10_ratio'] = consistency['driverId'].map(top_10_ratio)
        
        # Sidebar inputs for number of drivers to display
        n_consistent = st.sidebar.number_input("Number of consistent top finishers", min_value=1, value=5, step=1)
        n_fluctuating = st.sidebar.number_input("Number of fluctuating drivers", min_value=1, value=5, step=1)
        
        # Identify drivers with consistent top finishes:
        consistent_top = consistency[
            (consistency['avg_position'] < 10) &
            (consistency['position_std'] < consistency['position_std'].median())
        ].sort_values('position_std').head(int(n_consistent))
        
        # Identify drivers with fluctuating race results:
        fluctuating = consistency.nlargest(int(n_fluctuating), 'position_std')
        
        # Plotting: Scatter plot of average position vs. standard deviation
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(consistency['avg_position'], consistency['position_std'], c='gray', alpha=0.5, label='All Drivers')
        ax.scatter(consistent_top['avg_position'], consistent_top['position_std'], c='green', edgecolor='black', s=100, label='Consistent Top Finishers')
        ax.scatter(fluctuating['avg_position'], fluctuating['position_std'], c='red', edgecolor='black', s=100, label='Fluctuating Drivers')
        ax.set_xlabel("Average Finishing Position")
        ax.set_ylabel("Standard Deviation of Finishing Position")
        ax.set_title("Driver Consistency: Avg Position vs. Standard Deviation")
        ax.legend()
        plt.tight_layout()
        
        # Enhance the UI with formatted titles and use container width for tables
        outputs = []
        st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Driver Consistency Analysis</h3>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #2196F3;'>Consistent Top Finishers</h4>", unsafe_allow_html=True)
        st.info("Drivers who consistently finish in top positions with low variability in their finishing positions.")
        st.dataframe(consistent_top[['driverId', 'driver_name', 'avg_position', 'position_std', 'races', 'top_10_ratio']], use_container_width=True)
        
        st.markdown("<h4 style='text-align: center; color: #ff0000;'>Drivers with Fluctuating Results</h4>", unsafe_allow_html=True)
        st.info("Drivers with high variance in their finishing positions.")
        st.dataframe(fluctuating[['driverId', 'driver_name', 'avg_position', 'position_std', 'races', 'top_10_ratio']], use_container_width=True)
        
        st.pyplot(fig)
        
        outputs.append("Driver consistency analysis has been displayed above with enhanced formatting and both IDs and names.")
        return outputs, None
def lap_time_efficiency(df):
    outputs = []
    # Build mappings for constructor and driver names
    constructor_mapping = df[['constructorId', 'constructorRef']].drop_duplicates()
    driver_mapping = df[['driverId', 'driverRef']].drop_duplicates()

    # Load lap time data and merge with race, driver, and team info
    lap_df = load_data("master_lap_times.csv")
    lap_df = lap_df.merge(df[['raceId', 'driverId', 'constructorId']], on=['raceId', 'driverId'])

    # Calculate mean lap times per circuit for each driver and team
    avg_laps = lap_df.groupby(['circuitId', 'driverId', 'constructorId'])['milliseconds'].mean().reset_index()
    # Identify the fastest lap per circuit overall
    fastest_laps = lap_df.groupby('circuitId')['milliseconds'].min().reset_index()
    # Merge to compute an efficiency ratio (fastest lap / average lap time)
    efficiency = avg_laps.merge(fastest_laps, on='circuitId', suffixes=('_avg', '_fast'))
    efficiency['efficiency_ratio'] = efficiency['milliseconds_fast'] / efficiency['milliseconds_avg']

    # Overall team efficiency: average efficiency ratio across circuits
    team_efficiency = efficiency.groupby('constructorId')['efficiency_ratio'].mean().sort_values(ascending=False)

    # Create a pivot table to examine circuit-specific efficiency by team
    team_circuit_performance = efficiency.pivot_table(
        values='efficiency_ratio',
        index='constructorId',
        columns='circuitId',
        aggfunc='mean'
    ).fillna(0)
    
    # Let user choose the number of top teams and drivers to display
    n = st.sidebar.number_input("Number of Top Entries", min_value=1, value=5, step=1)

    # ------------------ First figure (will be overwritten) ------------------
    # Build a figure with two subplots: overall efficiency and circuit comparisons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Prepare data for bar chart with both IDs and names
    top_teams = team_efficiency.head(10).reset_index().merge(
        constructor_mapping, on='constructorId', how='left'
    )
    top_teams = top_teams.sort_values('efficiency_ratio', ascending=False)
    ax1.bar(
        top_teams['constructorId'].astype(str) + " - " + top_teams['constructorRef'],
        top_teams['efficiency_ratio'],
        color="steelblue",
        edgecolor="black"
    )
    ax1.set_title("Top 10 Teams by Overall Lap Time Efficiency", fontsize=16, weight="bold")
    ax1.set_xlabel("Constructor (ID - Name)", fontsize=12)
    ax1.set_ylabel("Efficiency Ratio", fontsize=12)
    ax1.tick_params(axis="x", rotation=45, labelsize=10)

    # Heatmap: Circuit-specific efficiency performance for top 5 teams
    sns.heatmap(
        team_circuit_performance.head(5),
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        ax=ax2,
        cbar_kws={"label": "Efficiency Ratio"},
        linewidths=0.5,
        linecolor="gray"
    )
    ax2.set_title("Circuit Efficiency Comparison for Top 5 Teams", fontsize=16, weight="bold")
    ax2.set_xlabel("Circuit ID", fontsize=12)
    ax2.set_ylabel("Constructor ID", fontsize=12)
    ax2.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()

    # Prepare dataframes for display with both IDs and names with dynamic top n entries
    top_teams_df = team_efficiency.head(n).to_frame().reset_index().merge(
        constructor_mapping, on="constructorId", how="left"
    )
    top_teams_df = top_teams_df[["constructorId", "constructorRef", "efficiency_ratio"]]

    driver_efficiency = efficiency.groupby('driverId')['efficiency_ratio'].mean().sort_values(ascending=False)
    top_drivers_df = driver_efficiency.head(n).to_frame().reset_index().merge(
        driver_mapping, on="driverId", how="left"
    )
    top_drivers_df = top_drivers_df[["driverId", "driverRef", "efficiency_ratio"]]

    # ------------------ Second figure (final figure to return) ------------------
    # Create a figure with three subplots: overall efficiency bar chart, circuit comparison heatmap, and lap time distribution boxplot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 18))

    # Plot 1: Bar Chart for Overall Efficiency by Team (Top 10)
    top_teams = team_efficiency.head(10).reset_index().merge(
        constructor_mapping, on='constructorId', how='left'
    )
    top_teams = top_teams.sort_values('efficiency_ratio', ascending=False)
    ax1.bar(
        top_teams['constructorId'].astype(str) + " - " + top_teams['constructorRef'],
        top_teams['efficiency_ratio'],
        color="steelblue",
        edgecolor="black"
    )
    ax1.set_title("Top 10 Teams by Overall Lap Time Efficiency", fontsize=16, weight="bold")
    ax1.set_xlabel("Constructor (ID - Name)", fontsize=12)
    ax1.set_ylabel("Efficiency Ratio", fontsize=12)
    ax1.tick_params(axis="x", rotation=45, labelsize=10)

    # Plot 2: Heatmap for Circuit-Specific Efficiency Comparison (Top 5 Teams)
    sns.heatmap(
        team_circuit_performance.head(5),
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        ax=ax2,
        cbar_kws={"label": "Efficiency Ratio"},
        linewidths=0.5,
        linecolor="gray"
    )
    ax2.set_title("Circuit Efficiency Comparison for Top 5 Teams", fontsize=16, weight="bold")
    ax2.set_xlabel("Circuit ID", fontsize=12)
    ax2.set_ylabel("Constructor ID", fontsize=12)
    ax2.tick_params(axis="both", which="major", labelsize=10)

    # Plot 3: Boxplot for Lap Time Distribution Across Circuits
    sns.boxplot(
        data=lap_df,
        x='circuitId',
        y='milliseconds',
        ax=ax3,
        palette='Set3'
    )
    ax3.set_title("Lap Time Distribution Across Circuits", fontsize=16, weight="bold")
    ax3.set_xlabel("Circuit ID", fontsize=12)
    ax3.set_ylabel("Lap Time (Milliseconds)", fontsize=12)
    ax3.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Explanation of the graphs:
    ts = ("The first graph (bar chart) presents the top 10 teams by overall lap time efficiency, "
          "displaying how each team’s average lap time compares to the fastest laps. "
          "Meanwhile, the heatmap and boxplot offer insights into circuit-specific performance and lap time variability, respectively, providing a comprehensive overview of efficiency across various tracks.")
    st.write(ts)

    # Display the dataframes with descriptions and full container width
    st.markdown("### Top Teams by Lap Time Efficiency")
    st.write("This table lists the top teams based on their overall lap time efficiency ratios.")
    st.dataframe(top_teams_df, use_container_width=True)

    st.markdown("### Top Drivers by Lap Time Efficiency")
    st.write("This table lists the top drivers based on their overall lap time efficiency ratios.")
    st.dataframe(top_drivers_df, use_container_width=True)

    return outputs, fig
def best_team_lineup(df, recent_years=3):
    outputs = []
    current_year = df['year'].max()
    recent_df = df[df['year'] >= current_year - recent_years]
    
    # Create driver mapping
    driver_mapping = df[['driverId', 'driverRef']].drop_duplicates().set_index('driverId')['driverRef']
    
    # Compute driver performance metrics in recent years
    points_per_race = recent_df.groupby('driverId').agg({
        'points': ['sum', 'mean'],
        'position': ['mean', 'std'],
        'raceId': 'count',
        'driverRef': 'first'
    }).round(2)
    points_per_race.columns = ['total_points', 'avg_points', 'avg_position', 'position_std', 'races', 'driver_name']
    driver_stats = points_per_race.reset_index()
    
    # Calculate podium and win ratios
    podiums = recent_df[recent_df['position'] <= 3].groupby('driverId').size()
    wins = recent_df[recent_df['position'] == 1].groupby('driverId').size()
    driver_stats['podium_ratio'] = (driver_stats['driverId'].map(podiums) / driver_stats['races']).fillna(0).round(3)
    driver_stats['win_ratio'] = (driver_stats['driverId'].map(wins) / driver_stats['races']).fillna(0).round(3)
    
    # Filter out drivers with insufficient races
    min_races = 10
    driver_stats = driver_stats[driver_stats['races'] >= min_races]
    
    # Normalize key metrics for comparison
    for col in ['avg_points', 'podium_ratio', 'win_ratio']:
        driver_stats[f'{col}_norm'] = (driver_stats[col] - driver_stats[col].min()) / (driver_stats[col].max() - driver_stats[col].min())
    
    driver_stats['consistency_score'] = 1 - (driver_stats['position_std'] / driver_stats['position_std'].max())
    
    # Calculate final score based on weighted metrics
    weights = {'avg_points_norm': 0.3, 'podium_ratio_norm': 0.25, 'win_ratio_norm': 0.25, 'consistency_score': 0.2}
    driver_stats['final_score'] = (
        driver_stats['avg_points_norm'] * weights['avg_points_norm'] +
        driver_stats['podium_ratio_norm'] * weights['podium_ratio_norm'] +
        driver_stats['win_ratio_norm'] * weights['win_ratio_norm'] +
        driver_stats['consistency_score'] * weights['consistency_score']
    )
    
    # Select best drivers based on final score
    best_drivers = driver_stats.nlargest(5, 'final_score')
    
    # Create dataframes for display
    lineup_df = best_drivers[['driverId', 'driver_name', 'avg_points', 'podium_ratio', 'win_ratio', 'consistency_score', 'final_score']]
    st.markdown("### Best Team Lineup (Based on Last {} Years)".format(recent_years))
    st.dataframe(lineup_df, use_container_width=True)
    
    recommended_df = best_drivers.iloc[:2][['driverId', 'driver_name', 'final_score']]
    st.markdown("### Recommended Driver Pairing")
    st.dataframe(recommended_df, use_container_width=True)
    
    # Generate supporting graph to visualize the final scores of top drivers
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([f"{row['driverId']} - {row['driver_name']}" for _, row in best_drivers.iterrows()], 
            best_drivers['final_score'], color='teal')
    ax.set_xlabel("Final Score")
    ax.set_title("Top 5 Drivers by Final Score")
    for index, value in enumerate(best_drivers['final_score']):
        ax.text(value, index, f"{value:.2f}", va='center', ha='left')
    plt.tight_layout()
    
    # Add supporting statistics and final recommendation
    top_pair = best_drivers.iloc[:2]
    combined_stats = {
        'Total Points': top_pair['total_points'].sum(),
        'Avg Points per Race': top_pair['avg_points'].mean(),
        'Combined Podiums': (top_pair['podium_ratio'] * top_pair['races']).sum(),
        'Combined Wins': (top_pair['win_ratio'] * top_pair['races']).sum(),
        'Avg Consistency': top_pair['consistency_score'].mean()
    }
    
    recommendation_text = f"""
    ### Final Team Lineup Recommendation
    
    Based on comprehensive analysis of the last {recent_years} years:
    
    **Recommended Driver Pairing:**
    1. {top_pair.iloc[0]['driver_name']} (ID: {top_pair.iloc[0]['driverId']})
       - Win Rate: {top_pair.iloc[0]['win_ratio']*100:.1f}%
       - Podium Rate: {top_pair.iloc[0]['podium_ratio']*100:.1f}%
       - Consistency Score: {top_pair.iloc[0]['consistency_score']:.3f}
    
    2. {top_pair.iloc[1]['driver_name']} (ID: {top_pair.iloc[1]['driverId']})
       - Win Rate: {top_pair.iloc[1]['win_ratio']*100:.1f}%
       - Podium Rate: {top_pair.iloc[1]['podium_ratio']*100:.1f}%
       - Consistency Score: {top_pair.iloc[1]['consistency_score']:.3f}
    
    **Combined Statistics:**
    - Total Points: {combined_stats['Total Points']:.0f}
    - Average Points per Race: {combined_stats['Avg Points per Race']:.1f}
    - Total Podiums: {combined_stats['Combined Podiums']:.0f}
    - Total Wins: {combined_stats['Combined Wins']:.0f}
    - Team Consistency Rating: {combined_stats['Avg Consistency']:.3f}
    
    This pairing provides an optimal balance of consistent high performance and race-winning capability.
    """
    
    st.markdown(recommendation_text)
    outputs.append("A horizontal bar chart is provided to visualize the final scores of the top drivers.")
    outputs.append(recommendation_text)
    
    return outputs, fig


# Define a simple Feedforward Neural Network using PyTorch
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training function for the neural network
def train_nn_model(model, X_train, y_train, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.to_numpy()).view(-1, 1)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Optional: Print progress every 10 epochs
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Prediction function for the neural network
def predict_nn_model(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).numpy().flatten()
    return predictions

def predictions_2025(df):
    outputs = []
    
    st.markdown("""
    ## 🏎️ Formula 1 2025 Season Predictions
    
    This analysis predicts driver performance for the 2025 season using three machine learning models:
    - 🌲 Random Forest
    - 🚀 XGBoost
    - 🧠 Neural Network
    
    The predictions are based on historical performance metrics, team strength, and driver experience.
    """)

    # Prepare data with enhanced features
    recent_df = df[df['year'] >= 2015].copy()
    
    # Feature engineering with clear naming
    recent_df['previous_season_points'] = recent_df.groupby('driverId')['points'].shift(1)
    recent_df['recent_form'] = recent_df.groupby('driverId')['points'].rolling(3).mean().reset_index(0, drop=True)
    recent_df['podium_finishes'] = (recent_df['position'] <= 3).astype(int)
    recent_df['race_wins'] = (recent_df['position'] == 1).astype(int)
    recent_df['did_not_finish'] = (recent_df['position'].isna()).astype(int)
    
    # Enhanced team strength calculation
    team_strength = recent_df.groupby(['year', 'constructorId']).agg({
        'points': 'mean',
        'podium_finishes': 'mean',
        'race_wins': 'mean'
    }).mean(axis=1).reset_index(name='team_strength')
    
    recent_df = recent_df.merge(team_strength, on=['year', 'constructorId'])
    recent_df['career_experience'] = recent_df.groupby('driverId')['raceId'].cumcount()
    
    # Prepare features with intuitive names
    features = [
        'grid',
        'previous_season_points',
        'recent_form',
        'team_strength',
        'career_experience',
        'podium_finishes',
        'race_wins',
        'did_not_finish'
    ]
    
    # Clean and scale data
    model_df = recent_df.dropna(subset=features + ['points'])
    X = model_df[features]
    y = model_df['points']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data with progress bar
    with st.spinner('Training models...'):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train models
        models = {
            '🌲 Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            '🚀 XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            '🧠 Neural Network': SimpleNN(input_size=X_train.shape[1])
        }
        
        # Train models with progress tracking
        progress_bar = st.progress(0)
        for idx, (name, model) in enumerate(models.items()):
            if name == '🧠 Neural Network':
                train_nn_model(model, X_train, y_train, epochs=100, lr=0.001)
            else:
                model.fit(X_train, y_train)
            progress_bar.progress((idx + 1) / len(models))

    # Make predictions for 2025
    latest_year = recent_df['year'].max()
    latest_data = recent_df[recent_df['year'] == latest_year].copy()
    
    predictions = []
    with st.spinner('Generating predictions...'):
        for driver_id in latest_data['driverId'].unique():
            driver_data = latest_data[latest_data['driverId'] == driver_id].iloc[0]
            
            pred_features = np.array([
                driver_data['grid'],
                driver_data['points'],
                driver_data['recent_form'],
                driver_data['team_strength'],
                driver_data['career_experience'] + 1,
                driver_data['podium_finishes'],
                driver_data['race_wins'],
                driver_data['did_not_finish']
            ]).reshape(1, -1)
            
            pred_features_scaled = scaler.transform(pred_features)
            
            model_predictions = {
                name: model.predict(pred_features_scaled)[0] if name != '🧠 Neural Network'
                else predict_nn_model(model, pred_features_scaled)[0]
                for name, model in models.items()
            }
            
            predictions.append({
                'driverId': driver_id,
                'driverRef': driver_data['driverRef'],
                **model_predictions,
                'avg_predicted_points': np.mean(list(model_predictions.values())),
                'current_points': driver_data['points']
            })
    
    # Create visualization
    pred_df = pd.DataFrame(predictions).sort_values('avg_predicted_points', ascending=False)
    
    # Display results with enhanced styling
    st.markdown("### 🏆 Top Predicted Drivers for 2025")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    top_driver = pred_df.iloc[0]
    
    with col1:
        st.metric("🥇 Predicted Champion", 
                 top_driver['driverRef'],
                 f"{top_driver['avg_predicted_points']:.0f} pts")
    
    with col2:
        point_change = top_driver['avg_predicted_points'] - top_driver['current_points']
        st.metric("📈 Points Change",
                 f"{point_change:+.0f}",
                 "from current season")
    
    with col3:
        prediction_confidence = 1 - (np.std([
            top_driver['🌲 Random Forest'],
            top_driver['🚀 XGBoost'],
            top_driver['🧠 Neural Network']
        ]) / top_driver['avg_predicted_points'])
        st.metric("🎯 Prediction Confidence",
                 f"{prediction_confidence:.1%}")
    
    # Display detailed predictions table
    st.markdown("### 📊 Detailed Predictions")
    st.dataframe(
        pred_df[['driverRef', 'avg_predicted_points', '🌲 Random Forest', 
                 '🚀 XGBoost', '🧠 Neural Network', 'current_points']],
        use_container_width=True
    )
    
    # Create visualizations
    fig = plt.figure(figsize=(15, 20))
    
    # Plot 1: Top 5 drivers prediction comparison
    ax1 = plt.subplot(3, 1, 1)
    top_5 = pred_df.head()
    x = np.arange(len(top_5))
    width = 0.25
    
    ax1.bar(x - width, top_5['🌲 Random Forest'], width, label='Random Forest')
    ax1.bar(x, top_5['🚀 XGBoost'], width, label='XGBoost')
    ax1.bar(x + width, top_5['🧠 Neural Network'], width, label='Neural Network')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_5['driverRef'], rotation=45)
    ax1.set_title('Model Comparison for Top 5 Drivers')
    ax1.legend()
    
    # Plot 2: Feature importance comparison
    ax2 = plt.subplot(3, 1, 2)
    feature_importance = pd.DataFrame({
        'feature': features,
        'Random Forest': models['🌲 Random Forest'].feature_importances_,
        'XGBoost': models['🚀 XGBoost'].feature_importances_
    }).melt(id_vars=['feature'], var_name='Model', value_name='Importance')
    
    sns.barplot(data=feature_importance, x='Importance', y='feature', hue='Model', ax=ax2)
    ax2.set_title('Feature Importance by Model')
    
    # Plot 3: Predicted vs Current Points
    ax3 = plt.subplot(3, 1, 3)  
    st.markdown("### Top 5 Predicted Drivers for 2025 (Averaged Predictions)")
    st.dataframe(
        pred_df[['driverRef', 'avg_predicted_points', '🌲 Random Forest', 
                     '🚀 XGBoost', '🧠 Neural Network', 'current_points']].rename(columns={
            'driverRef': 'Driver',
            'avg_predicted_points': 'Averaged Predicted Points',
            'rf_predicted_points': 'Random Forest Prediction',
            'xgb_predicted_points': 'XGBoost Prediction',
            'nn_predicted_points': 'Neural Network Prediction',
            'current_points': 'Current Points'
        }),
        use_container_width=True
    )
    
    st.markdown("### Visualizations")
    st.pyplot(fig)
    
    return outputs, None



def struggling_teams(df):
    """
    Analyze and predict teams likely to struggle in the 2025 season based on historical trends.
    
    Args:
        df (pd.DataFrame): DataFrame containing racing team data with columns 'year', 'points', 
                          'position', 'constructorId', 'constructorRef'
    
    Returns:
        tuple: (team_metrics DataFrame, matplotlib Figure)
    """
    current_year = df['year'].max()
    
    # Data preparation
    recent_df = df[df['year'] >= current_year - 4].copy()
    recent_df['year_weight'] = (1.5 ** (recent_df['year'] - (current_year - 4))) / (1.5 ** 4)
    recent_df['weighted_points'] = recent_df['points'] * recent_df['year_weight']
    
    # Calculate team metrics
    team_metrics = (recent_df.groupby('constructorId')
                   .agg({
                       'weighted_points': 'sum',
                       'points': ['mean', 'std', 'count'],
                       'position': ['mean', 'std'],
                       'constructorRef': 'first',
                       'year': 'nunique'
                   })
                   .round(2))
    
    team_metrics.columns = ['weighted_points', 'avg_points', 'points_std', 'race_count',
                           'avg_position', 'position_std', 'constructor_name', 'years_active']
    
    # Calculate yearly points and metrics
    yearly_points = recent_df.groupby(['year', 'constructorId'])['points'].sum().unstack()
    
    # Handle teams with insufficient data
    yearly_points_pct_change = yearly_points.pct_change()
    team_metrics['point_growth'] = yearly_points_pct_change.mean(axis=0).fillna(0)
    team_metrics['volatility'] = yearly_points_pct_change.std(axis=0).fillna(0)
    
    # Calculate scores
    team_metrics['development_score'] = team_metrics['point_growth']
    team_metrics['stability_score'] = 1 / (team_metrics['volatility'].clip(lower=0.001) + 1)
    team_metrics['participation_score'] = team_metrics['years_active'] / 4
    
    # Calculate consistency score safely
    team_metrics['consistency_score'] = (team_metrics['points_std'] / 
                                       team_metrics['avg_points'].replace(0, 0.001)).clip(upper=5)
    
    # Calculate risk score
    max_weighted_points = max(team_metrics['weighted_points'].max(), 1)
    team_metrics['risk_score'] = (
        (1 - team_metrics['weighted_points'] / max_weighted_points) * 0.35 +
        team_metrics['consistency_score'] * 0.20 +
        (1 - team_metrics['stability_score'].clip(0, 1)) * 0.20 +
        (team_metrics['development_score'] < 0).astype(float) * 0.15 +
        (1 - team_metrics['participation_score'].clip(0, 1)) * 0.10
    ).fillna(1.0)
    
    # Sort teams by risk
    team_metrics = team_metrics.sort_values('risk_score', ascending=False)
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    top_at_risk = team_metrics.head(5)
    
    # Risk Score Bar Plot
    sns.barplot(data=top_at_risk, x='constructor_name', y='risk_score', 
                palette='YlOrRd', ax=ax1)
    ax1.set_title('Top 5 Teams at Risk for 2025', fontsize=14, pad=20)
    ax1.set_xlabel('Constructor', fontsize=12)
    ax1.set_ylabel('Risk Score', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Performance Metrics Heatmap
    metrics_to_plot = ['weighted_points', 'stability_score', 'development_score', 'participation_score']
    sns.heatmap(top_at_risk[metrics_to_plot].T, annot=True, fmt='.2f', cmap='RdYlGn', 
                ax=ax2, xticklabels=top_at_risk['constructor_name'])
    ax2.set_title('Performance Metrics Breakdown', fontsize=14, pad=20)
    ax2.tick_params(axis='x', rotation=45)
    
    # Historical Points Trend
    for team_id in top_at_risk.index:
        team_data = recent_df[recent_df['constructorId'] == team_id]
        sns.lineplot(data=team_data, x='year', y='points', 
                    label=top_at_risk.loc[team_id, 'constructor_name'], 
                    ax=ax3, marker='o')
    ax3.set_title('Historical Points Trend', fontsize=14)
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Points', fontsize=12)
    ax3.legend(title='Constructor', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Streamlit report
    st.markdown("## 🏎️ Formula 1 Team Risk Analysis for 2025")
    st.subheader("Top 3 Teams at Risk - Detailed Analysis")
    
    for _, team in team_metrics.head(3).iterrows():
        risk_level = "🔴 High" if team['risk_score'] > 0.7 else "🟡 Moderate" if team['risk_score'] > 0.4 else "🟢 Low"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=team['constructor_name'],
                     value=f"Risk Score: {team['risk_score']:.3f}",
                     delta=risk_level)
        with col2:
            st.metric(label="Season Performance",
                     value=f"Avg Points: {team['avg_points']:.1f}",
                     delta=f"Stability: {team['stability_score']:.3f}")
            
        with st.expander(f"View detailed analysis for {team['constructor_name']}"):
            st.write("**Performance Indicators**")
            metrics_df = pd.DataFrame({
                'Metric': ['Development Trend', 'Point Stability', 'Participation', 'Risk Level'],
                'Value': [
                    '📉 Negative' if team['development_score'] < 0 else '📈 Positive',
                    '🔄 Stable' if team['stability_score'] >= 0.5 else '⚠️ Volatile',
                    f"{team['participation_score']*100:.0f}%",
                    risk_level
                ]
            })
            st.dataframe(metrics_df, hide_index=True)
    
    return team_metrics, fig


    
def driver_track_struggles(df):
    # Set page config for better UI
    st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>🏎️ Driver Track Performance Analysis</h1>", 
                unsafe_allow_html=True)
    
    # Create mappings
    driver_mapping = (df[['driverId', 'driverRef']]
                     .drop_duplicates()
                     .set_index('driverId')['driverRef'])
    circuit_mapping = (df[['circuitId', 'circuitRef']]
                      .drop_duplicates()
                      .set_index('circuitId')['circuitRef'])
    
    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Controls")
        selected_driver = st.selectbox(
            "Select Driver",
            sorted(df['driverRef'].unique()),
            help="Choose a driver to analyze their track performance"
        )
        # Add controls for number of top/bottom circuits
        n_circuits = st.slider(
            "Number of circuits to display",
            min_value=1,
            max_value=10,
            value=3,
            help="Select how many best/worst circuits to display"
        )
    
    # Handle case where driver has no data
    try:
        driver_id = df[df['driverRef'] == selected_driver]['driverId'].iloc[0]
        driver_data = df[df['driverId'] == driver_id].copy()
        
        if driver_data.empty:
            raise ValueError("No data available for selected driver")
            
    except (IndexError, ValueError) as e:
        st.error(f"Error: Unable to analyze {selected_driver}. No data available.")
        return pd.DataFrame(), plt.figure()
    
    # Calculate track performance metrics
    driver_circuit_stats = (driver_data.groupby('circuitId')
                          .agg({
                              'position': ['mean', 'std', 'count', 'min', 'max'],
                              'points': ['mean', 'sum'],
                              'grid': ['mean', 'min']
                          })
                          .round(2))
    
    driver_circuit_stats.columns = [
        'avg_position', 'position_std', 'races', 'best_finish', 'worst_finish',
        'avg_points', 'total_points', 'avg_grid', 'best_grid'
    ]
    
    # Add circuit names and handle missing values
    driver_circuit_stats['circuit_name'] = driver_circuit_stats.index.map(circuit_mapping)
    driver_circuit_stats = driver_circuit_stats.reset_index()
    driver_circuit_stats.fillna({'position_std': 0, 'avg_points': 0}, inplace=True)
    
    # Calculate performance score (lower is better)
    driver_circuit_stats['performance_score'] = (
        driver_circuit_stats['avg_position'] * 0.4 +
        driver_circuit_stats['position_std'].fillna(0) * 0.2 +
        (20 - driver_circuit_stats['avg_points'].fillna(0)) * 0.3 +
        driver_circuit_stats['avg_grid'] * 0.1
    ).clip(lower=0)
    
    # Identify best and worst tracks using user-selected n_circuits
    best_tracks = driver_circuit_stats.nsmallest(n_circuits, 'performance_score')
    worst_tracks = driver_circuit_stats.nlargest(n_circuits, 'performance_score')
    
    # Visualization
    fig = plt.figure(figsize=(14, 18))
    
    # Plot 1: Performance by Circuit
    ax1 = plt.subplot(3, 1, 1)
    sns.barplot(
        data=driver_circuit_stats.sort_values('performance_score'),
        x='circuit_name',
        y='performance_score',
        palette='RdYlGn_r',
        ax=ax1
    )
    ax1.set_title(f'Track Performance for {selected_driver}', pad=20, fontsize=16)
    ax1.set_xlabel('Circuit', fontsize=12)
    ax1.set_ylabel('Performance Score\n(Lower = Better)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Points Distribution
    ax2 = plt.subplot(3, 1, 2)
    sns.boxplot(
        data=driver_data,
        x='circuitRef',
        y='points',
        palette='muted',
        ax=ax2
    )
    ax2.set_title('Points Distribution by Circuit', pad=20, fontsize=16)
    ax2.set_xlabel('Circuit', fontsize=12)
    ax2.set_ylabel('Points', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Grid vs Finish Position
    ax3 = plt.subplot(3, 1, 3)
    selected_circuits = pd.concat([best_tracks, worst_tracks])['circuitId']
    circuit_data = driver_data[driver_data['circuitId'].isin(selected_circuits)]
    
    sns.scatterplot(
        data=circuit_data,
        x='grid',
        y='position',
        hue='circuitRef',
        style='circuitRef',
        s=150,
        alpha=0.7,
        ax=ax3
    )
    ax3.set_title(f'Grid vs Finish Position\n(Top {n_circuits} Best and Worst Circuits)', pad=20, fontsize=16)
    ax3.set_xlabel('Grid Position', fontsize=12)
    ax3.set_ylabel('Finish Position', fontsize=12)
    ax3.invert_yaxis()
    ax3.legend(title='Circuit', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    
    # UI Layout
    st.pyplot(fig)
    
    # Detailed Statistics
    st.markdown(f"<h3 style='margin-top: 20px;'>🏁 Performance Analysis for {selected_driver}</h3>", 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🟢 Top {n_circuits} Best Performing Circuits")
        st.dataframe(
            best_tracks[[
                'circuit_name', 'avg_position', 'best_finish', 'avg_points',
                'total_points', 'races'
            ]].rename(columns={
                'circuit_name': 'Circuit',
                'avg_position': 'Avg Pos',
                'best_finish': 'Best',
                'avg_points': 'Avg Pts',
                'total_points': 'Total Pts',
                'races': 'Races'
            }).style.format({
                'Avg Pos': '{:.1f}',
                'Avg Pts': '{:.1f}',
                'Total Pts': '{:.0f}'
            }),
            use_container_width=True
        )
    
    with col2:
        st.markdown(f"### 🔴 Top {n_circuits} Most Challenging Circuits")
        st.dataframe(
            worst_tracks[[
                'circuit_name', 'avg_position', 'worst_finish', 'avg_points',
                'total_points', 'races'
            ]].rename(columns={
                'circuit_name': 'Circuit',
                'avg_position': 'Avg Pos',
                'worst_finish': 'Worst',
                'avg_points': 'Avg Pts',
                'total_points': 'Total Pts',
                'races': 'Races'
            }).style.format({
                'Avg Pos': '{:.1f}',
                'Avg Pts': '{:.1f}',
                'Total Pts': '{:.0f}'
            }),
            use_container_width=True
        )
    
    # Performance Insights
    with st.expander("📊 Detailed Performance Insights", expanded=True):
        avg_finish = driver_circuit_stats['avg_position'].mean()
        best_circuit = best_tracks.iloc[0]
        worst_circuit = worst_tracks.iloc[0]
        
        insights = f"""
        **Overall Statistics:**
        - Average Finish Position: {avg_finish:.2f}
        - Circuits Raced: {len(driver_circuit_stats)}
        
        **Strongest Circuit: {best_circuit['circuit_name']}**
        - Avg Position: {best_circuit['avg_position']:.2f}
        - Best Finish: {best_circuit['best_finish']:.0f}
        - Avg Points: {best_circuit['avg_points']:.2f}
        
        **Weakest Circuit: {worst_circuit['circuit_name']}**
        - Avg Position: {worst_circuit['avg_position']:.2f}
        - Worst Finish: {worst_circuit['worst_finish']:.0f}
        - Avg Points: {worst_circuit['avg_points']:.2f}
        """
        st.markdown(insights)
    
    return [], None

def championship_retention(df):
    """
    Analyze championship retention patterns and predict probability of back-to-back titles.
    
    Args:
        df (pd.DataFrame): Formula 1 race results dataframe
        
    Returns:
        tuple: (list of output strings, matplotlib figure)
    """
    st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>🏆 Championship Retention Analysis</h1>", 
                unsafe_allow_html=True)
    
    # Get season champions (driver with most points each year)
    champions = df.groupby(['year', 'driverId'])['points'].sum().reset_index()
    yearly_champions = champions.loc[champions.groupby('year')['points'].idxmax()]
    yearly_champions = yearly_champions.sort_values('year')
    
    # Add driver names
    driver_mapping = df[['driverId', 'driverRef']].drop_duplicates().set_index('driverId')['driverRef']
    yearly_champions['driverRef'] = yearly_champions['driverId'].map(driver_mapping)
    
    # Calculate retention statistics
    total_years = len(yearly_champions) - 1  # Subtract 1 since we can't check retention for last year
    retained = 0
    retention_years = []
    
    for i in range(len(yearly_champions)-1):
        current_champ = yearly_champions.iloc[i]
        next_champ = yearly_champions.iloc[i+1]
        if current_champ['driverId'] == next_champ['driverId']:
            retained += 1
            retention_years.append(current_champ['year'])
    
    retention_prob = retained / total_years if total_years > 0 else 0
    
    # Calculate streak statistics
    current_streak = 1
    longest_streak = 1
    streak_holder = None
    current_holder = None
    
    for i in range(1, len(yearly_champions)):
        if yearly_champions.iloc[i]['driverId'] == yearly_champions.iloc[i-1]['driverId']:
            current_streak += 1
            if current_streak > longest_streak:
                longest_streak = current_streak
                streak_holder = yearly_champions.iloc[i]['driverRef']
        else:
            current_streak = 1
            
    current_champion = yearly_champions.iloc[-1]['driverRef']
    
    # Create visualization with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Championship retention timeline
    retention_data = []
    for i in range(len(yearly_champions)-1):
        year = yearly_champions.iloc[i]['year']
        retained = yearly_champions.iloc[i]['driverId'] == yearly_champions.iloc[i+1]['driverId']
        retention_data.append({'year': year, 'retained': retained})
    
    retention_df = pd.DataFrame(retention_data)
    
    sns.barplot(
        data=retention_df,
        x='year',
        y='retained',
        ax=ax1,
        color='skyblue'
    )
    ax1.set_title('Championship Retention by Year', fontsize=14, pad=20)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Title Retained', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Recent champions win distribution
    recent_years = 10
    recent_champions = yearly_champions[yearly_champions['year'] >= yearly_champions['year'].max() - recent_years]
    champion_counts = recent_champions['driverRef'].value_counts()
    
    sns.barplot(
        x=champion_counts.values,
        y=champion_counts.index,
        ax=ax2,
        palette='viridis'
    )
    ax2.set_title(f'Championships Won (Last {recent_years} Years)', fontsize=14, pad=20)
    ax2.set_xlabel('Number of Championships', fontsize=12)
    ax2.set_ylabel('Driver', fontsize=12)
    
    plt.tight_layout()
    
    # Display statistics
    st.markdown("### 📊 Championship Retention Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Retention Probability",
            f"{retention_prob:.1%}",
            "historical average"
        )
    
    with col2:
        st.metric(
            "Longest Streak",
            f"{longest_streak} years",
            f"by {streak_holder}" if streak_holder else None
        )
    
    with col3:
        st.metric(
            "Current Champion",
            current_champion
        )
    
    # Display detailed analysis
    st.markdown("### 🔍 Detailed Analysis")
    
    with st.expander("View Historical Data"):
        st.dataframe(
            yearly_champions[['year', 'driverRef', 'points']],
            use_container_width=True
        )
    
    # Calculate recent trends
    recent_retention = sum(retention_df['retained'].tail(5)) / 5
    
    analysis_text = f"""
    ### 📈 Key Findings
    
    - Historical retention rate: **{retention_prob:.1%}**
    - Recent retention rate (last 5 years): **{recent_retention:.1%}**
    - Longest championship streak: **{longest_streak}** years ({streak_holder})
    - Current champion: **{current_champion}**
    
    Based on historical patterns, the probability of {current_champion} retaining the title is:
    **{max(retention_prob, recent_retention):.1%}**
    
    {"⚠️ Note: Recent retention rate is higher than historical average, suggesting increased dominance." if recent_retention > retention_prob else ""}
    """
    
    st.markdown(analysis_text)
    
    return [analysis_text], fig

def champion_age_trends(df):
    """
    Analyze trends in Formula 1 champion ages over time.
    
    Args:
        df (pd.DataFrame): Formula 1 race results dataframe
        
    Returns:
        tuple: (list of output strings, matplotlib figure)
    """
    outputs = []
    
    # Load and merge driver data with race results
    drivers_df = load_data("drivers_clean.csv")
    champions = df[df['position'] == 1].copy()
    champions = champions.merge(
        drivers_df, 
        on='driverId', 
        how='left',
        suffixes=('', '_y')
    )
    
    # Calculate champion ages more accurately considering month/day of birth
    champions['dob'] = pd.to_datetime(champions['dob'], errors='coerce')
    champions['race_date'] = pd.to_datetime(champions['year'].astype(str) + '-12-31')  # Use end of year as approximate race date
    champions['age'] = (champions['race_date'] - champions['dob']).dt.days / 365.25
    
    # Filter out unrealistic ages and group by decade
    champions = champions[champions['age'].between(18, 60)]
    champions['decade'] = (champions['year'] // 10) * 10
    # Calculate decade statistics and display in Streamlit
    decade_stats = champions.groupby('decade').agg({
        'age': ['mean', 'min', 'max', 'count']
    }).round(1)
    
    st.markdown("### 📊 Decade-wise Statistics")
    st.dataframe(
        decade_stats.rename(
            columns={'age': ''},
            level=0
        ).rename(
            columns={
                'mean': 'Average Age',
                'min': 'Youngest',
                'max': 'Oldest',
                'count': 'Champions'
            },
            level=1
        ),
        use_container_width=True
    )
    
    # Calculate age statistics
    age_stats = {
        'average': champions['age'].mean(),
        'youngest': champions['age'].min(),
        'oldest': champions['age'].max(),
        'median': champions['age'].median(),
        'std': champions['age'].std()
    }
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Age trend over time with regression line
    sns.regplot(
        x='year', 
        y='age', 
        data=champions,
        scatter_kws={'alpha':0.5, 's':100},
        line_kws={'color':'red'},
        ax=ax1
    )
    ax1.set_title("F1 Champion Age Trends Over Time", fontsize=14, pad=20)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Champion Age", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Age distribution
    sns.histplot(
        data=champions,
        x='age',
        bins=20,
        ax=ax2,
        color='skyblue',
        edgecolor='black'
    )
    ax2.axvline(
        age_stats['average'], 
        color='red', 
        linestyle='--', 
        label=f"Mean Age: {age_stats['average']:.1f}"
    )
    # Plot 3: Age ranges by decade
    decade_data = champions.groupby('decade')['age'].agg(['min', 'max', 'mean']).reset_index()
    sns.barplot(
        data=decade_data,
        x='decade',
        y='mean',
        ax=ax3,
        color='lightgreen',
        alpha=0.6
    )
    # Add age range annotations
    for idx, row in decade_data.iterrows():
        ax3.vlines(idx, row['min'], row['max'], color='darkgreen', linewidth=2)
        ax3.text(idx, row['max'], f"{row['max']:.1f}", ha='center', va='bottom')
        ax3.text(idx, row['min'], f"{row['min']:.1f}", ha='center', va='top')
    
    ax3.set_title("Championship Age Ranges by Decade", fontsize=14, pad=20)
    ax3.set_xlabel("Decade", fontsize=12)
    ax3.set_ylabel("Age Range (Years)", fontsize=12)
    
    plt.tight_layout()
    
    # Format output statistics with decade information
    ax2.legend()
    
    plt.tight_layout()
    
    # Format output statistics
    outputs.append(f"""
    F1 Champion Age Statistics:
    --------------------------
    Average Age: {age_stats['average']:.1f} years
    Median Age: {age_stats['median']:.1f} years
    Youngest Champion: {age_stats['youngest']:.0f} years
    Oldest Champion: {age_stats['oldest']:.0f} years
    Standard Deviation: {age_stats['std']:.1f} years
    """)
    
    # Calculate and add recent trends
    recent_years = 10
    recent_champions = champions[champions['year'] >= champions['year'].max() - recent_years]
    recent_avg = recent_champions['age'].mean()
    
    outputs.append(f"""
    Recent Trends (Last {recent_years} Years):
    ----------------------------------------
    Average Age: {recent_avg:.1f} years
    Trend: {'Younger' if recent_avg < age_stats['average'] else 'Older'} than historical average
    """)
    
    return outputs, fig

def predict_future_team(driver_name, df):
    outputs = []
    try:
        # Validate input data
        driver_df = df[df['driverRef'] == driver_name]
        if driver_df.empty:
            st.error(f"No data found for driver: {driver_name}")
            return [], None

        # Data preparation
        driver_df = driver_df.sort_values('year')
        latest_year = driver_df['year'].max()
        
        # Calculate team history and performance metrics
        team_history = driver_df.groupby(['year', 'constructorRef']).agg({
            'points': 'sum',
            'position': ['mean', 'min'],
            'raceId': 'count'
        }).reset_index()
        
        # Calculate current team performance
        current_team_data = driver_df[driver_df['year'] == latest_year]
        current_team = current_team_data.groupby('constructorRef')['points'].sum().idxmax()
        current_performance = current_team_data['points'].sum()

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot 1: Points progression over years
        sns.lineplot(data=driver_df, x='year', y='points', ax=ax1, marker='o')
        ax1.set_title(f'Points Progression - {driver_name}', pad=20)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Points')
        ax1.grid(True)

        # Plot 2: Team durations
        team_durations = driver_df.groupby('constructorRef')['year'].nunique()
        sns.barplot(x=team_durations.values, y=team_durations.index, ax=ax2)
        ax2.set_title('Years with Each Team', pad=20)
        ax2.set_xlabel('Number of Years')
        
        # Plot 3: Performance by team
        team_avg_points = driver_df.groupby('constructorRef')['points'].mean()
        sns.barplot(x=team_avg_points.values, y=team_avg_points.index, ax=ax3)
        ax3.set_title('Average Points by Team', pad=20)
        ax3.set_xlabel('Average Points')

        plt.tight_layout()

        # Calculate prediction metrics
        recent_trend = driver_df.groupby('year')['points'].sum().pct_change().tail(3).mean()
        years_at_current = len(driver_df[driver_df['constructorRef'] == current_team])
        avg_team_duration = driver_df.groupby('constructorRef')['year'].nunique().mean()

        # Identify potential teams
        all_teams = df[df['year'] == latest_year].groupby('constructorRef')['points'].sum()
        potential_teams = all_teams[all_teams > current_performance].nlargest(3)

        # Generate insights
        st.markdown(f"### Analysis for {driver_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Team", current_team)
            st.metric("Years with Current Team", years_at_current)
        with col2:
            st.metric("Performance Trend", f"{recent_trend:.1%}")
            st.metric("Average Team Duration", f"{avg_team_duration:.1f} years")

        # Prediction and recommendations
        change_likelihood = "High" if (
            recent_trend < -0.1 or 
            years_at_current > avg_team_duration or 
            current_performance < all_teams.median()
        ) else "Low"

        st.markdown(f"### Team Change Prediction")
        st.write(f"Likelihood of team change: **{change_likelihood}**")
        
        if potential_teams.any():
            st.markdown("### Potential Future Teams")
            for team, points in potential_teams.items():
                st.write(f"- {team}: {points:.0f} points in {latest_year}")
        
        return outputs, fig

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return [], None

def predictive_model(df):
    """
    Creates a logistic regression model to predict podium finishes (top 3 positions)
    based on features like grid position and average points per driver.
    Forecasts outcomes for the entire dataset.

    Args:
        df (pd.DataFrame): Input DataFrame with columns 'position', 'points', 'grid', and 'driverId'.

    Returns:
        tuple:
            - dict: Contains 'accuracy', 'confusion_matrix', 'model', 'importance', and 'predictions'.
            - matplotlib.figure.Figure: Figure with feature importance and confusion matrix plots.
    """
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        df = df.copy()

        # Validate required columns
        required_columns = ['position', 'points', 'grid', 'driverId']
        if not all(col in df.columns for col in required_columns):
            return {"error": "Missing required columns"}, None

        # Create target variable (podium: 1 if position <= 3, else 0)
        df['podium'] = (df['position'] <= 3).astype(int)

        # Calculate average points and grid position per driver
        df['points_avg'] = df.groupby('driverId')['points'].transform('mean')
        df['grid_avg'] = df.groupby('driverId')['grid'].transform('mean')

        # Select features and target, handle missing data
        features = ['grid', 'points_avg', 'grid_avg']
        data = df[features + ['podium']].dropna()

        # Check for sufficient data (minimum 100 samples for reliable modeling)
        if len(data) < 100:
            return {"error": "Insufficient data for modeling"}, None

        # Prepare features (X) and target (y)
        X = data[features]
        y = data['podium']

        # Split data into training and test sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train logistic regression model
        model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_pred_test = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        cm = confusion_matrix(y_test, y_pred_test)

        # Make predictions for the entire dataset
        y_pred_all = model.predict(X)

        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot feature importance (absolute value of coefficients)
        importance = pd.DataFrame(
            {'feature': features, 'importance': abs(model.coef_[0])}
        )
        sns.barplot(data=importance, x='feature', y='importance', ax=ax1)
        ax1.set_title('Feature Importance')
        ax1.tick_params(axis='x', rotation=45)

        # Plot confusion matrix for the test set
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title(f'Confusion Matrix (Test Set)\nAccuracy: {accuracy:.2%}')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')

        # Adjust layout for better visualization
        plt.tight_layout()

        # Return results, including predictions for all data
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "model": model,
            "importance": importance,
            "predictions": y_pred_all  # Predictions for the entire dataset
        }, fig

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}, None

# Streamlit UI
st.title("Formula 1 Data Analysis Dashboard")

# List of drivers for selection
drivers = df['driverRef'].unique().tolist()

st.sidebar.markdown("## 🎛 Select Analysis")
analysis = st.sidebar.selectbox(
    "",
    [
        "Driver & Constructor Performance",
        "Qualifying vs. Race Performance",
        "Pit Stop Strategies",
        "Head-to-Head Driver Analysis",
        "Hypothetical Driver Swap",
        "Driver Team Network",
        "Team Performance Comparison",
        "Driver Consistency",
        "Lap Time Efficiency",
        "Best Team Lineup",
        "2025 Predictions",
        "Struggling Teams",
        "Driver Track Struggles",
        "Championship Retention",
        "Champion Age Trends",
        "Predict Future Team",
        "Predictive Model"
    ]
)

# Display analysis based on selection
if analysis == "Driver & Constructor Performance":
    st.header("Driver and Constructor Performance")
    st.write("View the top drivers by win ratio and constructors by number of wins.")
    text_outputs, figure = driver_constructor_performance(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Qualifying vs. Race Performance":
    st.header("Qualifying vs. Race Performance")
    st.write("Compare starting grid positions to final race positions.")
    text_outputs, figure = qualifying_vs_race_performance(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Pit Stop Strategies":
    st.header("Pit Stop Strategies")
    st.write("Analyze the impact of pit stops on race outcomes.")
    text_outputs, figure = analyze_pit_stop_strategies(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Head-to-Head Driver Analysis":
    st.header("Head-to-Head Driver Analysis")
    st.write("Compare drivers' performances in the same races.")
    text_outputs, figure = head_to_head_analysis(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Hypothetical Driver Swap":
    st.header("Hypothetical Driver Swap")
    st.write("Simulate swapping two drivers between teams.")
    driver1 = st.sidebar.selectbox("Driver 1", drivers, index=drivers.index("hamilton") if "hamilton" in drivers else 0)
    driver2 = st.sidebar.selectbox("Driver 2", drivers, index=drivers.index("massa") if "massa" in drivers else 1)
    if st.sidebar.button("Run Simulation"):
        text_outputs, figure = hypothetical_driver_swap(df, driver1, driver2)
        for text in text_outputs:
            st.write(text)
        if figure:
            st.pyplot(figure)

elif analysis == "Driver Team Network":
    st.header("Driver Team Network")
    st.write("Visualize driver movements between teams.")
    text_outputs, figure = driver_team_network(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Team Performance Comparison":
    st.header("Team Performance Comparison")
    st.write("Compare team success rates overall and by circuit.")
    text_outputs, figure = team_performance_comparison(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Driver Consistency":
    st.header("Driver Consistency")
    st.write("Identify drivers with consistent performance.")
    text_outputs, figure = driver_consistency(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Lap Time Efficiency":
    st.header("Lap Time Efficiency")
    st.write("Compare lap time efficiency across teams and drivers.")
    text_outputs, figure = lap_time_efficiency(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Best Team Lineup":
    st.header("Best Team Lineup")
    st.write("Select the optimal driver lineup based on recent performance.")
    text_outputs, figure = best_team_lineup(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "2025 Predictions":
    st.header("2025 Predictions")
    st.write("Predict top drivers and constructors for 2025.")
    text_outputs, figure = predictions_2025(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Struggling Teams":
    st.header("Struggling Teams")
    st.write("Identify teams likely to struggle in 2025.")
    text_outputs, figure = struggling_teams(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Driver Track Struggles":
    st.header("Driver Track Struggles")
    st.write("Find circuits where drivers excel or struggle.")
    text_outputs, figure = driver_track_struggles(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Championship Retention":
    st.header("Championship Retention")
    st.write("Analyze the likelihood of champions retaining their titles.")
    text_outputs, figure = championship_retention(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Champion Age Trends":
    st.header("Champion Age Trends")
    st.write("Examine trends in champion ages over time.")
    text_outputs, figure = champion_age_trends(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)

elif analysis == "Predict Future Team":
    st.header("Predict Future Team")
    st.write("Predict a driver’s next team based on past transitions.")
    driver_name = st.sidebar.selectbox("Driver", drivers, index=drivers.index("hamilton") if "hamilton" in drivers else 0)
    if st.sidebar.button("Predict"):
        text_outputs, figure = predict_future_team(driver_name , df)
        for text in text_outputs:
            st.write(text)
        if figure:
            st.pyplot(figure)

elif analysis == "Predictive Model":
    st.header("Predictive Model")
    st.write("Predict podium finishes based on grid position.")
    text_outputs, figure = predictive_model(df)
    for text in text_outputs:
        st.write(text)
    if figure:
        st.pyplot(figure)