Motivation
Improve the visual clarity and richness of the exploratory data analysis so per-dataset patterns are easier to compare and interpret.
Add categorical breakdowns and clearer palettes to address the request to "make it more visualised".
Keep the existing cleaning/imputation pipeline while surfacing more informative charts for downstream review.
Description
Updated analysis_pipeline.py to expand the per-dataset overview to a 3x3 layout and increased figure sizing via plot_dataset_overview(df, title); added gender and background count plots and applied palettes and custom colors for improved contrast.
Expanded the comparison view to a 2x3 layout in plot_comparison(df) and added personal_work and course breakdowns, histogram styling (element="step"), violin spreads, and consistent palette usage for clear dataset contrast.
Kept existing data preparation helpers (normalize_columns, clean_dataset, impute_dataset, validate_dataset) and plotting helpers (plot_success, plot_hist, plot_violin, plot_quartiles), and adjusted run_pipeline to call the new overview and comparison functions; the modified file is analysis_pipeline.py.
Testing
No automated tests were executed because the real dataset files were not available in the environment.
The code changes were committed locally and the plotting functions were exercised conceptually, but visual verification requires running run_pipeline with the actual Excel files (--dataset1/--dataset2).
