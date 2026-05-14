"""
Category Distribution — Before & After Filtering to Top 23
Uses the bottom-level (leaf) category from category_name.
───────────────────────────────────────────────────────────
Usage:
    python category_distribution.py
    python category_distribution.py --file train.tsv
"""

import pandas as pd
import argparse, sys, os

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="train.csv")
args = parser.parse_args()

if not os.path.exists(args.file):
    print(f"Error: '{args.file}' not found.")
    sys.exit(1)

# Auto-detect separator
with open(args.file, "r", encoding="utf-8") as f:
    sep = "\t" if "\t" in f.readline() else ","

df = pd.read_csv(args.file, sep=sep)

# Find category column
cat_col = next((c for c in df.columns if "category" in c.lower()), None)
if not cat_col:
    print(f"Error: No category column found. Columns: {list(df.columns)}")
    sys.exit(1)

# Extract the BOTTOM-LEVEL (leaf) category
df["leaf_category"] = df[cat_col].fillna("No label").apply(lambda x: x.split("/")[-1])

# ── Before filtering ──────────────────────────────────────
all_counts = df["leaf_category"].value_counts()
print("=" * 55)
print(f" BEFORE FILTERING — {len(df):,} listings, {len(all_counts)} leaf categories")
print("=" * 55)
for cat, count in all_counts.items():
    pct = count / len(df) * 100
    print(f"  {cat:<30} {count:>7,}  ({pct:5.1f}%)")

# ── Filter to top 23 ─────────────────────────────────────
top_23 = all_counts.head(23).index.tolist()
df_filtered = df[df["leaf_category"].isin(top_23)]
filtered_counts = df_filtered["leaf_category"].value_counts()

dropped = len(df) - len(df_filtered)
dropped_cats = len(all_counts) - len(filtered_counts)

print(f"\n{'=' * 55}")
print(f" AFTER FILTERING — {len(df_filtered):,} listings, {len(filtered_counts)} categories")
print(f" Dropped: {dropped:,} listings from {dropped_cats} categories")
print("=" * 55)
for cat, count in filtered_counts.items():
    pct = count / len(df_filtered) * 100
    print(f"  {cat:<30} {count:>7,}  ({pct:5.1f}%)")

print(f"\n{'─' * 55}")
print(f" Summary: {len(df):,} → {len(df_filtered):,} listings ({dropped:,} removed, {dropped / len(df) * 100:.1f}%)")