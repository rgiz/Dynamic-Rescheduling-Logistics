## Data Source Structure (confirmed)

- **df_cleaned** – one row per element of every trip (granular). Optimiser _does not_ read this directly.
- **df_trips** – one row per **trip** with `start_loc`, `end_loc`, `start_time`, `duration_min`, mileage. Acts as the **atomic unit** the optimiser may re‑assign.
- **df_routes** – one row per **route** (one driver’s planned day), aggregating the trips it already owns.
- **center_coordinates.csv** – mapping `location_id → (X, Y)` used to pre‑compute the straight‑line distance/time matrices.

> **Invariant**: trips are indivisible; re‑scheduling only swaps whole trips between routes. Driver ↔ route is 1‑to‑1.
