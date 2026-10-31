# P3 · Ward Pulse — public data (participant-facing)

Civic complaint records of a large Indian city (Bengaluru BBMP), plus ward boundaries and ward attributes.

## Files
- `grievances_public_wards.csv` — 556,561 complaints from 156 wards, Feb 2020 – Jun 2025:
  `complaint_id, category_group, category_raw, sub_category, grievance_time, ward_id, status`.
- `ward_features.csv` — all 198 wards: population & households (Census 2011), area, road length, lakes, parks, playgrounds, schools, police/fire stations, bus stops & routes, streetlights, zone, constituencies, centroid lat/lon.
- `wards.geojson` — 198 ward polygons (WGS84), property `ward_id`.
- `prediction_grid_template.csv` — the 40 target wards x 18 months (2024-01..2025-06) x 6 category groups you must predict (4,320 rows). These wards have NO complaint records in the public file.
- `sample_submission.csv` — `ward_id,month,category_group,predicted_count`.

## Category Groups (6 Total)
1. `electrical`: Streetlights, power supply, transformers, high tension wires.
2. `solid_waste`: Garbage blackspots, door-to-door collection, dumping, segregation.
3. `roads`: Potholes, asphalt resurfacing, footpaths, road cutting, dividers.
4. `trees`: Fallen trees, dangerous branches, trimming, storm damage.
5. `animals_health`: Stray animal management, public health, mosquito control.
6. `other`: Drainage, encroachment, unauthorized hoardings, general civic issues.
