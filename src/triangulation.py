import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from pathlib import Path

# Load the trips CSV (ensure path is correct from your notebook location)
df = pd.read_csv("../data/processed/trips.csv")  # use relative path if running from notebooks/

# Extract source-destination-distance triples
df_pairs = df[['source_center', 'destination_center', 'segment_osrm_distance']].copy()
df_pairs = df_pairs[df_pairs['source_center'] != df_pairs['destination_center']]  # skip self-pairs

# Aggregate: median distance between source-destination pairs
pair_medians = df_pairs.groupby(['source_center', 'destination_center'])['segment_osrm_distance'].median().reset_index()

# Generate location index mapping
locations = sorted(set(pair_medians['source_center']).union(set(pair_medians['destination_center'])))
location_idx = {loc: i for i, loc in enumerate(locations)}
n = len(locations)

# Build symmetric distance matrix
distance_matrix = np.full((n, n), np.nan)
for _, row in pair_medians.iterrows():
    i = location_idx[row['source_center']]
    j = location_idx[row['destination_center']]
    distance_matrix[i, j] = row['segment_osrm_distance']
    distance_matrix[j, i] = row['segment_osrm_distance']

# Fill diagonal and replace NaNs
np.fill_diagonal(distance_matrix, 0)
distance_matrix = np.nan_to_num(distance_matrix, nan=np.nanmax(distance_matrix))

# Apply MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(distance_matrix)

# Save coordinates
df_coords = pd.DataFrame(coords, columns=['x', 'y'])
df_coords['center_id'] = locations
output_path = Path("../data/processed/center_coordinates.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df_coords.to_csv(output_path, index=False)

print(f"Triangulation complete. Coordinates saved to {output_path}.")
