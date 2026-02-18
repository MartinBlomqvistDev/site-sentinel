"""
Site Sentinel — shared Python package.

Contains the core logic for the near-miss prediction pipeline:
  - site_sentinel.data.parser       — DFS Viewer CSV parsing
  - site_sentinel.features.engineering — kinematic and interaction feature computation
  - site_sentinel.features.targets  — binary target variable creation
  - site_sentinel.vision.transform  — UTM → pixel coordinate projection
  - site_sentinel.config            — YAML config loading
  - site_sentinel.logging_utils     — project-wide logger factory
"""
