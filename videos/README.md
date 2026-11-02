# Drop folder

Put clips in `in/`, run the detector, read the results from `out/`.

    docker compose run --rm detect

Each clip produces `out/<name>.mp4` with boxes drawn and `out/<name>.csv` with
one row per detection: frame, time_s, conf, x1, y1, x2, y2. Coordinates are in
the source frame's pixels, whatever `--input-scale` was used.

Both folders are kept in git but their contents are not.
