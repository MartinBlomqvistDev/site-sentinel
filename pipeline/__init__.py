# pipeline/ — ML pipeline scripts for Site Sentinel.
#
# Run scripts in order:
#   01_find_events     → rank highest-risk sessions in the dataset
#   02_build_dataset   → feature-engineer all sessions → master CSV
#   03a/b/c            → baseline model comparisons (XGBoost, LSTM, TCN)
#   04_train_rf        → final dual-target Random Forest
#   05_calibrate_camera → RANSAC homography for UTM → pixel projection
#   06_render_video    → produce annotated demo video
