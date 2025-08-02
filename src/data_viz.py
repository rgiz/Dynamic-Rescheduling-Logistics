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
