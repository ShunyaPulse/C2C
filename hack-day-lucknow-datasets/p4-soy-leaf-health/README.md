# P4 · Soy Leaf Doctor — public data (participant-facing)

Field photographs of soybean leaves from Indian farms, 6 classes (source not disclosed).

Files
- `images/` — 2,079 JPEGs, centre-cropped to square and resized to 384x384 (EXIF removed).
- `train_labels.csv` — image_id,label.
- `sample_submission.csv` — image_id,label + one probability column per class (p_<class>).

Classes: Caterpillar_Semilooper_Pest, Frog_Leaf_Eye, Healthy, Mosaic, Rust, Spectoria_Brown_Spot.
The hidden set comes from different field-visit days than the public images.
