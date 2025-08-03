"""Central place for column‑name constants so tests, preprocessing,
   and optimiser all stay in sync.

‼️  **Edit here once** if the raw CSV schema changes.  All downstream code
    (including tests) should import from this module instead of hard‑coding
    strings.  """

# ────────────────────────────────────────────────────────────────────────────
# Identifiers
# ────────────────────────────────────────────────────────────────────────────

TRIP_ID   = "trip_uuid"            # unique trip identifier in all tables
ROUTE_ID  = "route_schedule_uuid"  # unique route / driver‑day identifier

# ────────────────────────────────────────────────────────────────────────────
# Element‑level columns in df_cleaned (one row per trip element)
# ────────────────────────────────────────────────────────────────────────────

START_LOC = "source_center"         # integer / str location id of segment start
END_LOC   = "destination_center"    # location id of segment end
START_TS  = "od_start_time"         # planned / actual start timestamp
END_TS    = "od_end_time"           # planned / actual end timestamp
DURATION  = "segment_actual_time"   # minutes for this segment (int)
MILEAGE   = "segment_osrm_distance" # km for this segment (float)

# ────────────────────────────────────────────────────────────────────────────
# Aggregated columns produced during ETL (df_trips, df_routes)
# ────────────────────────────────────────────────────────────────────────────

ROUTE_START_TS = "route_start_time"   # earliest segment start per route
ROUTE_END_TS   = "route_end_time"     # latest segment end per route
ROUTE_TOTAL_TIME = "route_total_time"
ROUTE_TOTAL_DIST = "route_total_distance"