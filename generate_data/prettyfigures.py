# import math
# from pathlib import Path
# import re
# import pandas as pd
# import matplotlib.pyplot as plt
# import roi_label_map

# regions = roi_label_map.ROI_labels.values()
# regions = [r for r in regions if r != "background"]

# # # # def plot_diff_vs_age_subplots(csv_paths, figsize=(16, 10), max_regions_legend=10, save_path=None):
# # # #     n = len(csv_paths)
# # # #     ncols = math.ceil(math.sqrt(n))
# # # #     nrows = math.ceil(n / ncols)

# # # #     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
# # # #     axes_flat = axes.ravel()

# # # #     for ax, csv_path in zip(axes_flat, csv_paths):
# # # #         df = pd.read_csv(csv_path)

# # # #         # Coerce Age to numeric
# # # #         df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
# # # #         df = df.dropna(subset=["Age"])

# # # #         # Identify *_relative_volume_change_ columns
# # # #         for r in regions:
# # # #             diff_cols = [c for c in df.columns if c.endswith(f"_{r}_Intervened")] - [c for c in df.columns if c.endswith(f"_{r}_ISV")]
# # # #         # diff_cols = pd.to_numeric(df[diff_cols], errors="coerce").columns.tolist()
# # # #         if not diff_cols:
# # # #             ax.set_axis_off()
# # # #             ax.set_title(f"No *_relative_volume_change_ cols: {Path(csv_path).name}")
# # # #             continue

# # # #         # Group by Age and compute mean for each region
# # # #         grouped = df.groupby("Age")
# # # #         age_groups = sorted(grouped.groups.keys())

# # # #         for col in diff_cols:
# # # #             mean_vals = grouped[col].mean().reindex(age_groups)
# # # #             ax.scatter(age_groups, mean_vals.values, s=10, alpha=0.6, label=col.replace("_relative_volume_change_", ""))

# # # #         # Set titles/labels
# # # #         title = re.sub(r"_volume_comparison\.csv$", "", Path(csv_path).name)
# # # #         ax.set_title(title)
# # # #         ax.set_xlabel("Age")
# # # #         ax.set_ylabel("Mean Relative Volume Change (Intervened − ISV)")
# # # #         ax.grid(True, alpha=0.3)

# # # #         if len(diff_cols) <= max_regions_legend:
# # # #             ax.legend(fontsize=7, markerscale=2, frameon=True, loc="best")

# # # #     # Turn off any extra axes
# # # #     for j in range(n, len(axes_flat)):
# # # #         axes_flat[j].set_axis_off()
# # # #     # Global legend
# # # #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
# # # #                fancybox=True, shadow=True, ncol=5, fontsize='small')
# # # #     plt.tight_layout()
# # # #     if save_path:
# # # #         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
# # # #         plt.savefig(save_path, dpi=200, bbox_inches="tight")
# # # #         print(f"Saved to: {save_path}")

# # # #     plt.show()


# # # # # ==== Example usage ====


# # # # plot_diff_vs_age_subplots(csvs, figsize=(14, 10))
# # # import math
# # # from pathlib import Path
# # # import re
# # # import numpy as np
# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # import roi_label_map

# # # # regions list (optional; used only for legend name cleanup)
# # regions = [r for r in roi_label_map.ROI_labels.values() if r != "background"]

# # # def plot_diff_vs_age_subplots(csv_paths, figsize=(16, 10), max_regions_legend=10, save_path=None):
# # #     n = len(csv_paths)
# # #     ncols = math.ceil(math.sqrt(n))
# # #     nrows = math.ceil(n / ncols)

# # #     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
# # #     axes_flat = axes.ravel()

# # #     for ax, csv_path in zip(axes_flat, csv_paths):
# # #         df = pd.read_csv(csv_path)

# # #         # drop snippet-2 bottom summary row if present
# # #         if "subject" in df.columns:
# # #             df = df[df["subject"] != "__MAE__"].copy()

# # #         # Age numeric
# # #         df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
# # #         df = df.dropna(subset=["Age"])
# # #         if df.empty:
# # #             ax.set_axis_off()
# # #             ax.set_title(f"No valid Age rows: {Path(csv_path).name}")
# # #             continue

# # #         # pick already-computed change columns
# # #         change_cols = [c for c in df.columns
# # #                        if c.endswith("_relative_volume_change_")
# # #                        or c.endswith("_absolute_relative_volume_change_")
# # #                        or c.endswith("_diff")]  # keep this as fallback if you also saved *_diff
# # #         if not change_cols:
# # #             ax.set_axis_off()
# # #             ax.set_title(f"No change cols found: {Path(csv_path).name}")
# # #             continue

# # #         # coerce selected columns to numeric (in-place)
# # #         df[change_cols] = df[change_cols].apply(pd.to_numeric, errors="coerce")

# # #         # group by Age and order numerically
# # #         grouped = df.groupby("Age")
# # #         age_vals = np.array(sorted(grouped.groups.keys()))

# # #         # plot per-age MEANS (points only)
# # #         for col in change_cols:
# # #             mean_vals = grouped[col].mean().reindex(age_vals)
# # #             # nicer legend labels: strip suffixes and leading/trailing underscores
# # #             label = (col
# # #                      .replace("_relative_volume_change_", "")
# # #                      .replace("_absolute_relative_volume_change_", "")
# # #                      .replace("_diff", "")
# # #                      .strip("_"))
# # #             ax.scatter(age_vals, mean_vals.values, s=12, alpha=0.65, label=label)

# # #         # titles/labels
# # #         title = re.sub(r"_volume_comparison\.csv$", "", Path(csv_path).name)
# # #         ax.set_title(title)
# # #         ax.set_xlabel("Age")
# # #         ax.set_ylabel("Mean Change (Intervened − ISV)")
# # #         ax.grid(True, alpha=0.3)

# # #         # per-axes legend (cap size)
# # #         if len(change_cols) <= max_regions_legend:
# # #             ax.legend(fontsize=7, markerscale=2, frameon=True, loc="best")

# # #     # hide any unused axes
# # #     for j in range(n, len(axes_flat)):
# # #         axes_flat[j].set_axis_off()

# # #     plt.tight_layout()
# # #     if save_path:
# # #         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
# # #         plt.savefig(save_path, dpi=200, bbox_inches="tight")
# # #         print(f"Saved to: {save_path}")
# # #     plt.show()


# # # # ==== Example usage ====
# # # csvs = [
# # #     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_RPutamen_AGE_volume_comparison.csv",
# # #     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_LHippo_AGE_volume_comparison.csv",
# # #     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_LCortex_AGE_volume_comparison.csv",
# # #     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/chain/chain_LLV_AGE_volume_comparison.csv",
# # # ]

# # # plot_diff_vs_age_subplots(csvs, figsize=(14, 10))
# # import math
# # from pathlib import Path
# # import re
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # def plot_diff_vs_age_subplots(csv_paths, figsize=(16, 10), max_regions_legend=10, save_path=None):
# #     """
# #     Exactly replicate snippet-2 plots (grouped means by Age) for multiple CSVs, in subplots.
# #     """
# #     n = len(csv_paths)
# #     ncols = math.ceil(math.sqrt(n))
# #     nrows = math.ceil(n / ncols)
# #     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
# #     axes_flat = axes.ravel()

# #     for ax, csv_path in zip(axes_flat, csv_paths):
# #         df = pd.read_csv(csv_path)


# #         # --- compute difference per region
# #         for r in regions:        
# #             df[f"{r}_Intervened"] = pd.to_numeric(df[f"{r}_Intervened"], errors="coerce")
# #             df[f"{r}_ISV"] = pd.to_numeric(df[f"{r}_ISV"], errors="coerce")
# #             df[f"{r}_diff"] = df[f"{r}_Intervened"] - df[f"{r}_ISV"]


# #         # Coerce Age to numeric
# #         df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
# #         df = df.dropna(subset=["Age"])


# #         # Identify all *_diff columns exactly as snippet 2 does
# #         # diff_cols = [c for c in df.columns if c.endswith("_diff")]
# #         # Group by Age
# #         grouped = df.groupby("Age")
# #         age_vals = sorted(grouped.groups.keys())

# #         # Plot per-age mean differences (scatter only)
# #         for r in regions:
# #             mean_diff = grouped[f"{r}_diff"].mean()
# #             plt.scatter(age_vals, mean_diff.loc[age_vals], label=r)

# #         # Style identical to snippet 2
# #         title = re.sub(r"_volume_comparison\.csv$", "", Path(csv_path).name)
# #         ax.set_title(title)
# #         ax.set_xlabel("Age")
# #         ax.set_ylabel("Mean Volume Difference (Intervened − ISV)")
# #         ax.grid(True, alpha=0.3)

# #         # if len(diff_cols) <= max_regions_legend:
# #         ax.legend(fontsize=7, markerscale=2, frameon=True, loc="best")

# #     # Hide unused subplots
# #     for j in range(n, len(axes_flat)):
# #         axes_flat[j].set_axis_off()

# #     plt.tight_layout()
# #     if save_path:
# #         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
# #         plt.savefig(save_path, dpi=200, bbox_inches="tight")
# #         print(f"Saved figure to: {save_path}")

# #     plt.show()


# # # ==== Example usage ====
# # csvs = [
# #     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_RPutamen_AGE_volume_comparison.csv",
# #     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_LHippo_AGE_volume_comparison.csv",
# #     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_LCortex_AGE_volume_comparison.csv",
# #     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/chain/chain_LLV_AGE_volume_comparison.csv",
# # ]

# # plot_diff_vs_age_subplots(csvs, figsize=(14, 10))
# # --- paths
# path_isv = "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/none/ISV_only_Intensity_Norm_combined.csv"
# path_int = "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/Chain_rputamen_intensity_norm_combined.csv" #LLV_AGE_ISV_volcontrol/LLV_AGE_ISV_volcontrol_combined1.csv"
# out_csv  = "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_RPutamen_AGE_volume_comparison.csv" #LLV_AGE_ISV_volcontrol/LLV_AGE_ISV_volcontrol_volume_comparison.csv"

# # --- helpers
# def first5_digits(x):
#     """Take the first run of digits and keep its first 5 digits."""
#     if pd.isna(x):
#         return None
#     s = str(x)
#     m = re.search(r"\d+", s)
#     if not m:
#         return None
#     return m.group(0)[:5]

# def clean_cols(df):
#     df = df.copy()
#     df.columns = [c.strip() for c in df.columns]
#     if "subject" not in df.columns:
#         raise ValueError("CSV is missing 'subject' column.")
#     df["subject"] = df["subject"].astype(str).str.strip()
#     return df

# def coerce_numeric(series):
#     # remove thousand separators and coerce to float
#     return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


# # a = clean_cols(pd.read_csv(path_isv))
# # b = clean_cols(pd.read_csv(path_int))

# # # normalize subject to first 5 digits
# # a["subject_norm"] = a["subject"].apply(first5_digits)
# # b["subject_norm"] = b["subject"].apply(first5_digits)

# # # pick region columns BEFORE merge (intersection)
# # regions_a = [c for c in a.columns if c not in ("subject", "subject_norm")]
# # regions_b = [c for c in b.columns if c not in ("subject", "subject_norm")]
# # regions = [r for r in regions_a if r in regions_b]

# # # keep only subject_norm + regions (+ Effect in B)
# # if "Effect1" not in b.columns:
# #     raise RuntimeError("The intervened file is missing an 'Effect' column.")
# # a_sub = a[["subject", "subject_norm"] + regions].copy()
# # b_sub = b[["subject", "subject_norm", "Effect1"] + regions].copy()

# # merged = pd.merge(a_sub, b_sub, on="subject_norm", suffixes=("_ISV", "_Intervened"))
# merged = pd.read_csv(out_csv)  # to overwrite existing file
# print(merged.shape)
# merged = merged.dropna(subset=["Age"])
# print(merged.shape)

# # 1) Are there duplicate subject labels?
# dupes = merged["subject"].duplicated(keep=False)
# print("Duplicate subject rows:", dupes.sum())
# print(merged.loc[dupes, "subject"].value_counts().head(10))

# # 2) Is the index unique after set_index?
# tmp = merged.set_index("subject", drop=False)
# print("Index unique?", tmp.index.is_unique)

# # 3) Any weird whitespace / hidden chars?
# bad = merged["subject"][merged["subject"].str.contains(r"\s|^\s|\s$", regex=True, na=False)]
# print("Subjects with whitespace:", bad.nunique(), "examples:", bad.unique()[:5])

# # 4) Mixed dtypes or leading zeros?
# print(merged["subject"].head().tolist(), merged["subject"].dtype)
# print(merged["subject"].str.len().describe())       # if string-like
# print(pd.to_numeric(merged["subject"], errors="coerce").isna().sum(), "non-numeric")

# # merged = merged.reset_index(drop=True)
# # coerce numeric for each paired region column
# for r in regions:
#     merged[f"{r}_ISV"]        = coerce_numeric(merged[f"{r}_ISV"])
#     merged[f"{r}_Intervened"] = coerce_numeric(merged[f"{r}_Intervened"])


# merged = merged.set_index("subject")
# merged['Age'] = merged.loc[merged.index, "Age"].values

# # --- compute difference per region
# for r in regions:
#     merged[f"{r}_diff"] = merged[f"{r}_Intervened"] - merged[f"{r}_ISV"]

# # --- group by age and compute mean difference
# grouped = merged.groupby("Age")
# age_vals = sorted(grouped.groups.keys())

# plt.figure(figsize=(12, 6))
# for r in regions:
#     mean_diff = grouped[f"{r}_diff"].mean()
#     plt.scatter(age_vals, mean_diff.loc[age_vals], label=r)

# plt.xlabel("Age")
# plt.ylabel("Mean Volume Difference (Intervened - ISV)")
# plt.title("Average Volume Difference by Age Across Regions")
# plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
# plt.tight_layout()
# plt.grid(True)
# plt.show()

# # 1) Make Age numeric and drop rows without Age
# merged["Age"] = pd.to_numeric(merged["Age"], errors="coerce")
# merged = merged.dropna(subset=["Age"]).reset_index(drop=True)

# # 2) (Optional) If you *must* reattach Age by subject, make a unique map:
# age_map = (
#     merged.loc[merged["Age"].notna(), ["subject", "Age"]]
#           .drop_duplicates(subset=["subject"])
#           .set_index("subject")["Age"]
# )
# merged["Age"] = merged["subject"].map(age_map)  # length always matches

# # 3) Coerce ROI columns and compute diffs
# for r in regions:
#     merged[f"{r}_ISV"]        = coerce_numeric(merged.get(f"{r}_ISV"))
#     merged[f"{r}_Intervened"] = coerce_numeric(merged.get(f"{r}_Intervened"))
#     merged[f"{r}_diff"]       = merged[f"{r}_Intervened"] - merged[f"{r}_ISV"]

# # 4) Group and plot (exactly like snippet 2)
# grouped = merged.groupby("Age")
# age_vals = sorted(grouped.groups.keys())

# plt.figure(figsize=(12, 6))
# for r in regions:
#     col = f"{r}_diff"
#     if col in merged.columns:
#         mean_diff = grouped[col].mean()
#         plt.scatter(age_vals, mean_diff.loc[age_vals], label=r)

# plt.xlabel("Age")
# plt.ylabel("Mean Volume Difference (Intervened - ISV)")
# plt.title("Average Volume Difference by Age Across Regions")
# plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
# plt.tight_layout()
# plt.grid(True)
# plt.show()




# import re
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_volume_change_vs_age(csv_path, col_patterns=("absolute_relative_volume_change",
#                                                       "relative_volume_change",
#                                                       "_diff"),
#                               title_suffix=None):
#     """
#     Load a results CSV (from your merge pipeline) and plot per-age mean volume change
#     for each ROI as scatter points (no lines).
#     """
#     df = pd.read_csv(csv_path)

#     # Drop bottom summary row if present
#     if "subject" in df.columns:
#         df = df[df["subject"] != "__MAE__"].copy()

#     # Age numeric and valid rows only
#     df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
#     df = df.dropna(subset=["Age"])
#     if df.empty:
#         raise ValueError("No rows with valid Age.")

#     # Find change columns by pattern (first pattern that hits)
#     change_cols = []
#     for pat in col_patterns:
#         hits = [c for c in df.columns if pat in c]
#         if hits:
#             change_cols = hits
#             break
#     if not change_cols:
#         raise ValueError("No change columns found. "
#                          "Expected something like _absolute_relative_volume_change_, "
#                          "_relative_volume_change_, or _diff.")

#     # Coerce selected columns to numeric
#     df[change_cols] = df[change_cols].apply(pd.to_numeric, errors="coerce")


#     # Plot per-age MEANs as scatter points
#     plt.figure(figsize=(12, 6))
#     for col in change_cols:
#         # mean_vals = grouped[col].mean().reindex(age_vals)
#         # Make legend label cleaner by stripping suffixes/underscores
#         label = (col
#                  .replace("_absolute_relative_volume_change_", "")
#                  .replace("_relative_volume_change_", "")
#                  .replace("_diff", "")
#                  .strip("_"))
#         plt.scatter(df['Age'], df[col], s=14, alpha=0.7, label=label)

#     # Axis labels and title
#     if any("_diff" in c for c in change_cols):
#         ylab = "Mean Volume Difference (Intervened − ISV)"
#     else:
#         ylab = "Mean Relative Volume Change (Intervened − ISV)"
#     plt.xlabel("Age")
#     plt.ylabel(ylab)

#     base_title = re.sub(r"_volume_comparison\.csv$", "", str(csv_path).split("/")[-1])
#     plt.title(base_title if title_suffix is None else f"{base_title} — {title_suffix}")

#     plt.grid(True, alpha=0.3)
#     plt.legend(loc="best", fontsize=8, frameon=True)
#     plt.tight_layout()
#     plt.show()


# # ==== Example usage ====
# csv_path = "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_RPutamen_AGE_volume_accuracy.csv"
# plot_volume_change_vs_age(csv_path)


import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# def plot_volume_change_vs_age_subplots(csv_paths,
#                                        col_patterns=("absolute_relative_volume_change",
#                                                      "relative_volume_change",
#                                                      "_diff"),
#                                        figsize=(16, 10),
#                                        max_regions_legend=10,
#                                        save_path=None):
#     """
#     Create subplots (one per CSV) showing raw per-subject volume change vs Age for each ROI.
#     """
#     n = len(csv_paths)
#     if n == 0:
#         raise ValueError("No CSV files provided.")
#     ncols = math.ceil(math.sqrt(n))
#     nrows = math.ceil(n / ncols)

#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
#     axes_flat = axes.ravel()

#     for ax, csv_path in zip(axes_flat, csv_paths):
#         df = pd.read_csv(csv_path)

#         # Drop summary row if present
#         if "subject" in df.columns:
#             df = df[df["subject"] != "__MAE__"].copy()

#         # Age numeric
#         df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
#         df = df.dropna(subset=["Age"])
#         if df.empty:
#             ax.set_axis_off()
#             ax.set_title(f"No valid Age rows: {Path(csv_path).name}")
#             continue

#         # Find columns for volume change
#         change_cols = []
#         for pat in col_patterns:
#             hits = [c for c in df.columns if pat in c]
#             if hits:
#                 change_cols = hits
#                 break
#         if not change_cols:
#             ax.set_axis_off()
#             ax.set_title(f"No change cols: {Path(csv_path).name}")
#             continue

#         # Coerce numeric for these columns
#         df[change_cols] = df[change_cols].apply(pd.to_numeric, errors="coerce")

#         # Scatter raw points
#         for col in change_cols:
#             label = (col
#                      .replace("_absolute_relative_volume_change_", "")
#                      .replace("_relative_volume_change_", "")
#                      .replace("_diff", "")
#                      .strip("_"))
#             ax.scatter(df["Age"], df[col], s=10, alpha=0.5, label=label)

#         # Title and axes labels
#         title = re.sub(r"_volume_comparison\.csv$", "", Path(csv_path).name)
#         ax.set_title(title)
#         ax.set_xlabel("Age")
#         ax.set_ylabel("Volume Change (Intervened − ISV)")
#         ax.grid(True, alpha=0.3)
#         if len(change_cols) <= max_regions_legend:
#             ax.legend(fontsize=7, markerscale=2, frameon=True, loc="best")

#     # Turn off any unused subplots
#     for j in range(n, len(axes_flat)):
#         axes_flat[j].set_axis_off()

#     plt.tight_layout()
#     if save_path:
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=200, bbox_inches="tight")
#         print(f"Saved figure to: {save_path}")
#     plt.show()


# ###############################################################


# import math
# import re
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
# from matplotlib.lines import Line2D

# def _clean_label(col):
#     return (col
#             .replace("_absolute_relative_volume_change_", "")
#             .replace("_relative_volume_change_", "")
#             .replace("_diff", "")
#             .strip("_"))

# def plot_volume_change_vs_age_subplots(
#     csv_paths,
#     col_patterns=("absolute_relative_volume_change", "relative_volume_change", "_diff"),
#     figsize=(16, 10),
#     save_path=None
# ):
#     if not csv_paths:
#         raise ValueError("No CSV files provided.")

#     # 1) Scan all files to collect the union of region labels
#     all_regions = []
#     file_cols = []  # per-file list of matched columns
#     for csv_path in csv_paths:
#         df = pd.read_csv(csv_path)
#         if "subject" in df.columns:
#             df = df[df["subject"] != "__MAE__"].copy()
#         # find change cols by first matching pattern
#         change_cols = []
#         for pat in col_patterns:
#             hits = [c for c in df.columns if pat in c]
#             if hits:
#                 change_cols = hits
#                 break
#         file_cols.append(change_cols)
#         all_regions.extend(_clean_label(c) for c in change_cols)

#     # unique, stable ordering of labels
#     region_labels = sorted(set(all_regions))
#     if not region_labels:
#         raise ValueError("No change columns found in any file.")

#     # 2) Build a consistent style map (color + marker) per region
#     markers = ['o','s','^','D','P','X','v','<','>','h','*','+','x','1','2','3','4','p']
#     cmap = get_cmap('tab20', 20)
#     style_map = {}
#     for i, lab in enumerate(region_labels):
#         color = cmap(i % 20)
#         marker = markers[(i // 20) % len(markers)]  # switch marker after colors run out
#         style_map[lab] = dict(color=color, marker=marker)

#     # 3) Make subplots
#     n = len(csv_paths)
#     ncols = math.ceil(math.sqrt(n))
#     nrows = math.ceil(n / ncols)
#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
#     axes_flat = axes.ravel()

#     for ax, csv_path, change_cols in zip(axes_flat, csv_paths, file_cols):
#         df = pd.read_csv(csv_path)
#         if "subject" in df.columns:
#             df = df[df["subject"] != "__MAE__"].copy()

#         # Age numeric
#         df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
#         df = df.dropna(subset=["Age"])
#         if not change_cols or df.empty:
#             ax.set_axis_off()
#             title = Path(csv_path).name if df.empty else f"No change cols: {Path(csv_path).name}"
#             ax.set_title(title)
#             continue

#         # coerce numeric for the columns we’ll plot
#         df[change_cols] = df[change_cols].apply(pd.to_numeric, errors="coerce")

#         # scatter raw points with consistent styles
#         for col in change_cols:
#             lab = _clean_label(col)
#             st = style_map[lab]
#             ax.scatter(df["Age"], df[col], s=12, alpha=0.6, label=lab,
#                        color=st["color"], marker=st["marker"])

#         title = re.sub(r"_volume_comparison\.csv$", "", Path(csv_path).name)
#         ax.set_title(title)
#         ax.set_xlabel("Age")
#         ax.set_ylabel("Volume Change (Intervened − ISV)")
#         ax.grid(True, alpha=0.3)

#     # Turn off any unused axes
#     for j in range(n, len(axes_flat)):
#         axes_flat[j].set_axis_off()

#     # 4) Global legend on the right (covers all 32 regions)
#     legend_handles = [
#         Line2D([0],[0], linestyle='',
#                marker=style_map[lab]["marker"],
#                color=style_map[lab]["color"],
#                label=lab, markersize=6)
#         for lab in region_labels
#     ]
#     # leave space on right for the legend
#     plt.tight_layout(rect=[0, 0, 0.82, 1])
#     fig.legend(legend_handles, [h.get_label() for h in legend_handles],
#                loc='center left', bbox_to_anchor=(0.84, 0.5),
#                ncol=1, fontsize='small', frameon=True)

#     if save_path:
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=200, bbox_inches="tight")
#         print(f"Saved figure to: {save_path}")

#     plt.show()

# # ==== Example usage ====
# csvs = [
#     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_RPutamen_AGE_volume_accuracy.csv",
#     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_RAmy_AGE_volume_accuracy.csv",
#     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_Lhippocampus_AGE_volume_accuracy.csv",
#     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_Lcortex_AGE_volume_accuracy.csv",
#     "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/chain/chain_LLV_AGE_volume_accuracy.csv",
# ]

# plot_volume_change_vs_age_subplots(csvs, figsize=(14, 10))\




import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

# ---------- palette helpers ----------
def get_distinct_palette(n):
    """
    Return n visually distinct colors.
    Prefers Glasbey (colorcet), then distinctipy, else HSV fallback.
    """
    # Try Glasbey
    try:
        import colorcet as cc 
        # cc.glasbey is 256 colors; slice first n
        print(f"Using Glasbey palette for {n} colors.")
        return [cc.glasbey[i] for i in range(n)]
    except Exception:
        pass
    # Try distinctipy
    try:
        import distinctipy
        cols = distinctipy.get_colors(n, exclude_colors=[(1,1,1),(0,0,0)])
        return cols
    except Exception:
        pass
    # Fallback: evenly spaced HSV
    return [(plt.cm.hsv(i / max(n,1)))[:3] for i in range(n)]



MARKERS = ['o','s','^','D','P','X','v','<','>','h','*','+','x','1','2','3','4','p']

def build_style_map(labels):
    """
    Stable color+marker map for the provided label list.
    With Glasbey/distinctipy you likely won't need markers for 32,
    but we keep them in case you go beyond the palette size later.
    """
    labels = list(labels)
    n = len(labels)
    colors = get_distinct_palette(n)
    style = {}
    for i, lab in enumerate(labels):
        style[lab] = {
            "color": colors[i % len(colors)],
            "marker": MARKERS[(i // len(colors)) % len(MARKERS)],  # only changes if > palette size
        }
    return style

# ---------- label cleaner ----------
def _clean_label(col):
    return (col
            .replace("_absolute_relative_volume_change_", "")
            .replace("_relative_volume_change_", "")
            .replace("_diff", "")
            .strip("_"))

# ---------- main plotting ----------
def plot_volume_change_vs_age_subplots(
    csv_paths,
    subplot_titles=None,  # optional region names
    col_patterns=("absolute_relative_volume_change", "relative_volume_change", "_diff"),
    figsize=(16, 10),
    save_path=None
):
    if not csv_paths:
        raise ValueError("No CSV files provided.")

    # 1) Scan files to collect union of region labels
    all_regions = []
    file_cols = []  # per-file list of matched columns
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "subject" in df.columns:
            df = df[df["subject"] != "__MAE__"].copy()

        change_cols = []
        for pat in col_patterns:
            hits = [c for c in df.columns if pat in c]
            if hits:
                change_cols = hits
                break
        file_cols.append(change_cols)
        all_regions.extend(_clean_label(c) for c in change_cols)

    region_labels = sorted(set(all_regions))
    if not region_labels:
        raise ValueError("No change columns found in any file.")

    # 2) Build distinct, stable styles
    style_map = build_style_map(region_labels)

    # 3) Subplots grid
    n = len(csv_paths)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for idx, (ax, csv_path, change_cols) in enumerate(zip(axes_flat, csv_paths, file_cols)):
        df = pd.read_csv(csv_path)
        if "subject" in df.columns:
            df = df[df["subject"] != "__MAE__"].copy()

        # Age numeric
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df.dropna(subset=["Age"])
        if not change_cols or df.empty:
            ax.set_axis_off()
            title = Path(csv_path).name if df.empty else f"No change cols: {Path(csv_path).name}"
            ax.set_title(title)
            continue

        # Coerce numeric for plotted columns
        df[change_cols] = df[change_cols].apply(pd.to_numeric, errors="coerce")

        # Raw scatter per region with consistent styles
        for col in change_cols:
            lab = _clean_label(col)
            st = style_map[lab]
            ax.scatter(df["Age"], df[col], s=14, alpha=0.65,
                       color=st["color"], marker=st["marker"], label=lab)
            
                # use provided subplot title or fallback to filename
        if subplot_titles and idx < len(subplot_titles):
            title = subplot_titles[idx]
        else:
            title = re.sub(r"_volume_(change of ROI|accuracy)\.csv$", "", Path(csv_path).name, flags=re.I)
        ax.set_title(title)
        ax.set_xlabel("A (cause)")
        ax.set_ylabel("Volume Change (effect)")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    # 4) Global legend (won't crush subplots)
    handles = [
        Line2D([0],[0], linestyle='',
               marker=style_map[lab]["marker"],
               color=style_map[lab]["color"],
               label=lab, markersize=6)
        for lab in region_labels
    ]
    # leave space for legend on the right
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    fig.legend(handles, [h.get_label() for h in handles],
               loc='center left', bbox_to_anchor=(0.84, 0.5),
               ncol=1, fontsize='small', frameon=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()

# ==== Example usage ====

titles = [
    "Right Putamen",
    "Right Amygdala",
    "Left Hippocampus",
    "Left Cortex",
    "Left Lateral Ventricle"
]

csvs = [
    "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_RPutamen_AGE_volume_accuracy.csv",
    "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_RAmy_AGE_volume_accuracy.csv",
    "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_Lhippocampus_AGE_volume_accuracy.csv",
    "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_Lcortex_AGE_volume_accuracy.csv",
    "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/chain/chain_LLV_AGE_volume_accuracy.csv",
]

# plot_volume_change_vs_age_subplots(csvs, figsize=(14, 10), subplot_titles=titles, save_path='/home/eryn/SimBA/figures/SCAR/volume_change_vs_age_subplots_testingColours1.png')



import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

# ---------- palette helpers ----------
def get_sequential_palette(n):
    """
    Return n visually distinct sequential colors from a Matplotlib colormap.
    Uses 'viridis' which is perceptually uniform and publication-friendly.
    """
    cmap = plt.cm.get_cmap("Spectral", n)
    return [cmap(i) for i in range(n)]

def build_style_map(labels):
    """
    Assign colors and markers:
    - Left regions: circle
    - Right regions: triangle
    - Non-matched: square
    Colors are shared between left/right of the same region.
    """
    # Normalize to base region names
    base_regions = sorted(set(
        re.sub(r'^(left|right)\s+', '', lab, flags=re.I) for lab in labels
    ))
    colors = get_sequential_palette(len(base_regions))

    # Map base region -> color
    region_color_map = {reg: colors[i] for i, reg in enumerate(base_regions)}

    style = {}
    for lab in labels:
        base = re.sub(r'^(left|right)\s+', '', lab, flags=re.I)
        if "left" in lab.lower():
            marker = "^"
        elif "right" in lab.lower():
            marker = "s"
        else:
            marker = "*"
        style[lab] = {
            "color": region_color_map[base],
            "marker": marker,
            "base": base
        }
    return style

# ---------- label cleaner ----------
def _clean_label(col):
    return (col
            .replace("_absolute_relative_volume_change_", "")
            .replace("_relative_volume_change_", "")
            .replace("_diff", "")
            .strip("_"))

# ---------- main plotting ----------
def plot_volume_change_vs_age_subplots(
    csv_paths,
    subplot_titles=None,  # optional region names
    col_patterns=("absolute_relative_volume_change", "relative_volume_change", "_diff"),
    figsize=(16, 10),
    save_path=None
):
    if not csv_paths:
        raise ValueError("No CSV files provided.")

    # 1) Scan files to collect union of region labels
    all_regions = []
    file_cols = []  # per-file list of matched columns
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "subject" in df.columns:
            df = df[df["subject"] != "__MAE__"].copy()

        change_cols = []
        for pat in col_patterns:
            hits = [c for c in df.columns if pat in c]
            if hits:
                change_cols = hits
                break
        file_cols.append(change_cols)
        all_regions.extend(_clean_label(c) for c in change_cols)

    region_labels = sorted(set(all_regions))
    if not region_labels:
        raise ValueError("No change columns found in any file.")

    # 2) Build distinct, stable styles
    style_map = build_style_map(region_labels)

    # 3) Subplots grid
    n = len(csv_paths)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for idx, (ax, csv_path, change_cols) in enumerate(zip(axes_flat, csv_paths, file_cols)):
        df = pd.read_csv(csv_path)
        if "subject" in df.columns:
            df = df[df["subject"] != "__MAE__"].copy()

        # Age numeric
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df.dropna(subset=["Age"])
        if not change_cols or df.empty:
            ax.set_axis_off()
            title = Path(csv_path).name if df.empty else f"No change cols: {Path(csv_path).name}"
            ax.set_title(title)
            continue

        # Coerce numeric for plotted columns
        df[change_cols] = df[change_cols].apply(pd.to_numeric, errors="coerce")

        # Raw scatter per region with consistent styles
        for col in change_cols:
            lab = _clean_label(col)
            st = style_map[lab]
            ax.scatter(df["Age"], df[col], s=14, alpha=0.65,
                       color=st["color"], marker=st["marker"], label=lab)
            
        # use provided subplot title or fallback to filename
        if subplot_titles and idx < len(subplot_titles):
            title = subplot_titles[idx]
        else:
            title = re.sub(r"_volume_(change of ROI|accuracy)\.csv$", "", Path(csv_path).name, flags=re.I)
        ax.set_title(title)
        ax.set_xlabel("A (cause)")
        ax.set_ylabel("Volume Change (effect)")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    # # 4) Global legend
    # # Build legend grouped by base region
    # base_regions = sorted(set(st["base"] for st in style_map.values()))
    # handles = []
    # labels_out = []
    # handles.append(Line2D([0],[0], linestyle='', marker='s',
    #                           color='black', label=f"Left", markersize=6))
    # labels_out.append(f"Left")
    # handles.append(Line2D([0],[0], linestyle='', marker='^',
    #                           color='black', label=f"Right", markersize=6))
    # labels_out.append(f"Right")
    # for base in base_regions:
    #     color = style_map[[lab for lab in style_map if style_map[lab]["base"] == base][0]]["color"]
    #     # Add left/right markers if they exist
    #     if any("left" in lab.lower() for lab in style_map if style_map[lab]["base"] == base):
    #         handles.append(Line2D([0],[0], linestyle='', marker='o',
    #                               color=color, label=f"{base}", markersize=6))
    #         labels_out.append(f"{base}")
    #     # if any("right" in lab.lower() for lab in style_map if style_map[lab]["base"] == base):
    #     #     handles.append(Line2D([0],[0], linestyle='', marker='^',
    #     #                           color=color, label=f"Right {base}", markersize=6))
    #     #     labels_out.append(f"Right {base}")
    #     # If neither left/right, just square
    #     if not any(("left" in lab.lower() or "right" in lab.lower()) for lab in style_map if style_map[lab]["base"] == base):
    #         handles.append(Line2D([0],[0], linestyle='', marker='*',
    #                               color=color, label=base, markersize=6))
    #         labels_out.append(base)
    #     # add circle and triangle with any color for "Left" and "Right" legend entries
    # # order so that Left/Right markers appear first, then squares, then other regions
    # handles = handles[:2] + [h for h in handles[2:] if h.get_marker() == 's'] + [h for h in handles[2:] if h.get_marker() not in [ 's']]

    # plt.tight_layout(rect=[0, 0, 0.82, 1])
    # fig.legend(handles, labels_out,
    #            loc='center left', bbox_to_anchor=(0.84, 0.5),
    #            ncol=1, fontsize='small', frameon=True)
        # 4) Legends
    # Hemisphere legend (black/white markers only)
    hemisphere_handles = [
        Line2D([0],[0], linestyle='', marker='^', color='black',
               label="Left hemisphere", markersize=6),
        Line2D([0],[0], linestyle='', marker='s', color='black',
               label="Right hemisphere", markersize=6),
        Line2D([0],[0], linestyle='', marker='*', color='black',
               label="Unpaired region", markersize=6),
    ]

    # Region legend (colored circles)
    base_regions = sorted(set(st["base"] for st in style_map.values()))
    region_handles = [
        Line2D([0],[0], linestyle='', marker='o',
               color=style_map[[lab for lab in style_map if style_map[lab]["base"] == base][0]]["color"],
               label=base, markersize=6)
        for base in base_regions
    ]

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    # Place hemisphere legend on the right
    fig.legend(hemisphere_handles, [h.get_label() for h in hemisphere_handles],
               loc='center left', bbox_to_anchor=(0.76, 0.7),
               fontsize='small', frameon=True)
    # Place region legend below
    fig.legend(region_handles, [h.get_label() for h in region_handles],
               loc='center left', bbox_to_anchor=(0.76, 0.2),
               fontsize='small', frameon=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()

# ==== Example usage ====
titles = [
    "Right Putamen",
    "Right Amygdala",
    "Left Hippocampus",
    "Left Cortex",
    "Left Lateral Ventricle"
]
csvs = [
    "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_RPutamen_AGE_volume_accuracy.csv",
    "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_RAmy_AGE_volume_accuracy.csv",
    "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_Lhippocampus_AGE_volume_accuracy.csv",
    "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/addROIs/chain/chain_Lcortex_AGE_volume_accuracy.csv",
    "/home/eryn/SynthSegOutput/intensity_norm_isv1.25/chain/chain_LLV_AGE_volume_accuracy.csv",
]
plot_volume_change_vs_age_subplots(csvs, figsize=(14, 10), subplot_titles=titles)#, save_path='/home/eryn/SimBA/figures/SCAR/volume_change_vs_age_subplots_Spectral.png')

# get the slope of volume change vs age for each region and do pearsons correlation vs each region

# get slope of region volume change vs age using linear regression
from sklearn.linear_model import LinearRegression
def compute_slopes(csv_paths, col_patterns=("absolute_relative_volume_change",
                                            "relative_volume_change",
                                            "_diff")):
    slopes = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "subject" in df.columns:
            df = df[df["subject"] != "__MAE__"].copy()

        # Age numeric
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df.dropna(subset=["Age"])
        if df.empty:
            continue

        # Find change columns
        change_cols = []
        for pat in col_patterns:
            hits = [c for c in df.columns if pat in c]
            if hits:
                change_cols = hits
                break
        if not change_cols:
            continue

        # Coerce numeric
        df[change_cols] = df[change_cols].apply(pd.to_numeric, errors="coerce")

        # Compute slope for each region
        for col in change_cols:
            lab = _clean_label(col)
            X = df["Age"].values.reshape(-1, 1)
            y = df[col].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            slopes[lab] = slope
    return slopes

slopes = compute_slopes(csvs)
# for region, slope in slopes.items():
#     print(f"{region}: slope = {slope:.6f}")

# compute pearsons correlation between slopes of different regions
from scipy.stats import pearsonr
def compute_pearsons_correlation(slopes):   
    regions = list(slopes.keys())
    n = len(regions)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                r, _ = pearsonr([slopes[regions[i]]], [slopes[regions[j]]])
                corr_matrix[i, j] = r
    return regions, corr_matrix

regions, corr_matrix = compute_pearsons_correlation(slopes)
print("Pearson's correlation matrix between region slopes:")
print("Regions:", regions)
print(corr_matrix)    
