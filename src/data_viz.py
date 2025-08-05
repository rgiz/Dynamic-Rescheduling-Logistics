import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_trip_start_times(df_trips):
    df_trips['hour'] = df_trips['od_start_time'].dt.hour
    plt.figure(figsize=(10, 6))
    sns.histplot(df_trips['hour'], bins=24, kde=False)
    plt.title("Distribution of Trip Start Times by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Trips")
    plt.grid(True)
    plt.show()

def plot_trip_durations(df_trips):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_trips['trip_duration_minutes'], bins=50, kde=True)
    plt.title("Distribution of Trip Durations")
    plt.xlabel("Trip Duration (minutes)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_top_source_centers(df_trips, top_n=10):
    top_sources = df_trips['source_center'].value_counts().nlargest(top_n)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_sources.index, y=top_sources.values)
    plt.title(f"Top {top_n} Source Centers by Trip Count")
    plt.xticks(rotation=45)
    plt.xlabel("Source Center")
    plt.ylabel("Trip Count")
    plt.grid(True)
    plt.show()

def plot_trip_delay(df_trips):
    df_trips['delay_minutes'] = df_trips['actual_time'] - df_trips['osrm_time']
    plt.figure(figsize=(10, 6))
    sns.histplot(df_trips['delay_minutes'], bins=50, kde=True)
    plt.title("Distribution of Trip Delays (Actual - OSRM)")
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_trip_volume_by_day(df_trips):
    df_trips['date'] = df_trips['od_start_time'].dt.date
    volume = df_trips.groupby('date').size()
    plt.figure(figsize=(14, 6))
    volume.plot()
    plt.title("Trip Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Trips")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sla_breaches(df_trips, threshold_minutes):
    df_trips['delay_minutes'] = df_trips['actual_time'] - df_trips['osrm_time']
    breaches = df_trips[df_trips['delay_minutes'] > threshold_minutes]
    plt.figure(figsize=(10, 6))
    sns.histplot(breaches['delay_minutes'], bins=30, kde=True, color="red")
    plt.title(f"Trips with SLA Breaches (>{threshold_minutes} min delay)")
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Breach Count")
    plt.grid(True)
    plt.show()

def plot_geographic_distribution(df_trips):
    top_routes = df_trips.groupby(['source_center', 'destination_center']).size().reset_index(name='count')
    top_routes = top_routes.sort_values(by='count', ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_routes, x='count', y='source_center', hue='destination_center')
    plt.title("Top 20 Most Common Route Pairs")
    plt.xlabel("Trip Count")
    plt.ylabel("Source Center")
    plt.legend(title="Destination")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_route_total_times(df_routes, duty_limit_hours=12):
    """
    Visualize the distribution of route total times (shift durations) with 
    regulatory duty limit overlay.
    
    Parameters:
    -----------
    df_routes : pd.DataFrame
        Routes dataframe with route_shift_duration or route_total_time column
    duty_limit_hours : float
        Regulatory duty limit in hours (default 12)
    """
    
    # Use route_shift_duration if available, otherwise route_total_time
    time_col = 'route_shift_duration' if 'route_shift_duration' in df_routes.columns else 'route_total_time'
    
    # Convert to hours for better readability
    route_hours = df_routes[time_col] / 60.0
    duty_limit_minutes = duty_limit_hours * 60
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram with KDE
    sns.histplot(route_hours, bins=50, kde=True, ax=ax1, alpha=0.7, color='skyblue')
    ax1.axvline(x=duty_limit_hours, color='red', linestyle='--', linewidth=2, 
                label=f'{duty_limit_hours}h Duty Limit')
    ax1.set_xlabel('Route Duration (hours)')
    ax1.set_ylabel('Number of Routes')
    ax1.set_title('Distribution of Route Total Times')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    sns.boxplot(y=route_hours, ax=ax2, color='lightcoral')
    ax2.axhline(y=duty_limit_hours, color='red', linestyle='--', linewidth=2,
                label=f'{duty_limit_hours}h Duty Limit')
    ax2.set_ylabel('Route Duration (hours)')
    ax2.set_title('Route Duration Box Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Statistics
    over_limit = (df_routes[time_col] > duty_limit_minutes).sum()
    total_routes = len(df_routes)
    pct_over_limit = (over_limit / total_routes) * 100
    
    # Add statistics text
    stats_text = f"""Statistics:
    Total Routes: {total_routes:,}
    Over {duty_limit_hours}h limit: {over_limit:,} ({pct_over_limit:.1f}%)
    Mean: {route_hours.mean():.1f}h
    Median: {route_hours.median():.1f}h
    Max: {route_hours.max():.1f}h"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'total_routes': total_routes,
        'over_limit_count': over_limit,
        'over_limit_percentage': pct_over_limit,
        'mean_hours': route_hours.mean(),
        'median_hours': route_hours.median(),
        'max_hours': route_hours.max()
    }

def plot_trip_total_times(df_trips, duty_limit_hours=12):
    """
    Visualize the distribution of individual trip durations to test hypothesis 
    that 1 trip = 1 day's work for drivers.
    
    Parameters:
    -----------
    df_trips : pd.DataFrame
        Trips dataframe with trip_duration_minutes column
    duty_limit_hours : float
        Daily duty limit in hours for comparison (default 12)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Use trip_duration_minutes column
    time_col = 'trip_duration_minutes'
    
    if time_col not in df_trips.columns:
        print(f"Warning: {time_col} not found. Available columns: {list(df_trips.columns)}")
        return None
    
    # Convert to hours for better readability
    trip_hours = df_trips[time_col] / 60.0
    duty_limit_minutes = duty_limit_hours * 60
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram with KDE
    sns.histplot(trip_hours, bins=100, kde=True, ax=ax1, alpha=0.7, color='lightgreen')
    ax1.axvline(x=duty_limit_hours, color='red', linestyle='--', linewidth=2, 
                label=f'{duty_limit_hours}h Daily Duty Limit')
    ax1.axvline(x=8, color='orange', linestyle=':', linewidth=2, 
                label='8h Standard Work Day')
    ax1.set_xlabel('Trip Duration (hours)')
    ax1.set_ylabel('Number of Trips')
    ax1.set_title('Distribution of Individual Trip Durations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    sns.boxplot(y=trip_hours, ax=ax2, color='lightgreen')
    ax2.axhline(y=duty_limit_hours, color='red', linestyle='--', linewidth=2,
                label=f'{duty_limit_hours}h Daily Duty Limit')
    ax2.axhline(y=8, color='orange', linestyle=':', linewidth=2,
                label='8h Standard Work Day')
    ax2.set_ylabel('Trip Duration (hours)')
    ax2.set_title('Trip Duration Box Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Statistics
    near_full_day = ((df_trips[time_col] >= 6*60) & (df_trips[time_col] <= 12*60)).sum()  # 6-12 hours
    over_limit = (df_trips[time_col] > duty_limit_minutes).sum()
    total_trips = len(df_trips)
    pct_near_full_day = (near_full_day / total_trips) * 100
    pct_over_limit = (over_limit / total_trips) * 100
    
    # Add statistics text
    stats_text = f"""Trip Duration Statistics:
    Total Trips: {total_trips:,}
    6-12h trips (≈full day): {near_full_day:,} ({pct_near_full_day:.1f}%)
    Over {duty_limit_hours}h limit: {over_limit:,} ({pct_over_limit:.1f}%)
    Mean: {trip_hours.mean():.1f}h
    Median: {trip_hours.median():.1f}h
    Max: {trip_hours.max():.1f}h
    Min: {trip_hours.min():.1f}h"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'total_trips': total_trips,
        'near_full_day_count': near_full_day,
        'near_full_day_percentage': pct_near_full_day,
        'over_limit_count': over_limit,
        'over_limit_percentage': pct_over_limit,
        'mean_hours': trip_hours.mean(),
        'median_hours': trip_hours.median(),
        'max_hours': trip_hours.max(),
        'min_hours': trip_hours.min()
    }
