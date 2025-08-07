def get_travel_time_trips(from_location, to_location, 
                             distance_matrix=None, location_to_index=None, 
                             fallback_hours=8):
    """
    Get travel time between locations using trip-based matrix.
    Returns time in minutes, or fallback if pair doesn't exist.
    
    Args:
        from_location: Source location ID
        to_location: Destination location ID  
        distance_matrix: Preloaded time matrix (optional)
        location_to_index: Preloaded location index mapping (optional)
        fallback_hours: Default time if no data (hours)
    
    Returns:
        Travel time in minutes
    """
    
    # Load data if not provided
    if distance_matrix is None or location_to_index is None:
        import numpy as np
        from pathlib import Path
        
        dist_file = Path("data/dist_matrix.npz")  # Using original filename
        if not dist_file.exists():
            print(f"Warning: Distance matrix not found, using fallback")
            return fallback_hours * 60
        
        dist_data = np.load(str(dist_file), allow_pickle=True)
        distance_matrix = dist_data['time']
        location_ids = dist_data['ids']
        location_to_index = {str(loc): i for i, loc in enumerate(location_ids)}
    
    # Handle same location
    if from_location == to_location:
        return 0.0
    
    # Look up in matrix
    if from_location in location_to_index and to_location in location_to_index:
        from_idx = location_to_index[from_location]
        to_idx = location_to_index[to_location]
        
        time_minutes = distance_matrix[from_idx, to_idx]
        
        if not np.isnan(time_minutes) and time_minutes > 0:
            return float(time_minutes)
    
    # No data found, use fallback
    print(f"Warning: No trip data for {from_location} -> {to_location}, using {fallback_hours}h fallback")
    return fallback_hours * 60

# Example usage:
# travel_time = get_travel_time_trips("IND000000AAL", "IND000000AAQ")
# print(f"Travel time: {travel_time:.1f} minutes ({travel_time/60:.1f} hours)")
