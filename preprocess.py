"""
preprocess.py
Reads all NHS Hospital Admissions Excel files and produces
static/nhs_merged.csv with standardised columns.
"""

import os
import re
import glob
import pandas as pd
import numpy as np

BASE_DIR   = os.path.join(os.path.dirname(__file__), "NHS Hospital Admissions")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "static")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "nhs_merged.csv")

def skiprows_for_year(year: int) -> int:
    if 1998 <= year <= 2004: return 3
    if 2005 <= year <= 2006: return 10
    if 2007 <= year <= 2011: return 15
    if year == 2012:         return 17
    if year == 2013:         return 15
    if year == 2014:         return 9
    if 2015 <= year <= 2019: return 10
    if 2020 <= year <= 2022: return 9
    if year == 2023:         return 0
    raise ValueError(f"No skiprow mapping for year {year}")

CODE_RE = re.compile(r'^([A-Z]\d{2}(?:-[A-Z]?\d{2,3})?(?:\+)?)\s+(.+)$')

def find_col(cols, *substrings, case=False):
    """Return name of first column matching ANY of the substrings (case-insensitive by default)."""
    for col in cols:
        haystack = col if case else col.lower()
        for s in substrings:
            needle = s if case else s.lower()
            if needle in haystack:
                return col
    return None

def to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(',', '', regex=False).str.strip()
    s = s.replace(['*', '-', 'c', 'nan', 'NaN', 'None', ''], np.nan)
    return pd.to_numeric(s, errors='coerce').fillna(0)

def read_file(filepath: str, year: int) -> pd.DataFrame:
    skip = skiprows_for_year(year)
    is_old = year <= 2011

    if is_old:
        df = pd.read_excel(filepath, sheet_name=0, header=skip,
                           engine='xlrd')
    else:
        # Sheet name varies slightly across years — find it by fuzzy match
        xl = pd.ExcelFile(filepath, engine='openpyxl')
        sheet = next(
            (s for s in xl.sheet_names
             if 'summary' in s.lower() and 'primary' in s.lower()),
            None
        )
        if sheet is None:
            raise ValueError(f"Cannot find Primary/Summary sheet. Available: {xl.sheet_names}")
        df = pd.read_excel(filepath, sheet_name=sheet,
                           header=skip, engine='openpyxl')

    # Strip \n from ALL column names, then strip whitespace
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]

    # First column (index 0) is always the code/merged column
    first_col = df.columns[0]
    df = df.dropna(subset=[first_col])
    df = df[~df[first_col].astype(str).str.strip().str.lower().str.startswith('total')]
    df = df[df[first_col].astype(str).str.strip() != '']

    cols = list(df.columns)

    if is_old:
        # Code and description merged in first column
        raw = df[first_col].astype(str).str.strip()
        codes = []
        descs = []
        for val in raw:
            m = CODE_RE.match(val)
            if m:
                codes.append(m.group(1))
                descs.append(m.group(2).strip())
            else:
                codes.append(np.nan)
                descs.append(val)
        df['diagnosis_code'] = codes
        df['diagnosis_desc'] = descs
        # Drop rows where code couldn't be parsed
        df = df[df['diagnosis_code'].notna()]
    else:
        df['diagnosis_code'] = df[first_col].astype(str).str.strip()
        unnamed1 = 'Unnamed: 1'
        if unnamed1 in cols:
            df['diagnosis_desc'] = df[unnamed1].astype(str).str.strip()
        else:
            # Fallback: description in second column
            df['diagnosis_desc'] = df[cols[1]].astype(str).str.strip()

    fce_col = find_col(cols, 'finished consultant')
    df['fce'] = to_numeric(df[fce_col]) if fce_col else 0

    if is_old and year <= 2004:
        # No separate admissions column — use FCE
        df['admissions'] = df['fce']
    else:
        adm_col = find_col(cols, 'finished admission', 'admissions')
        df['admissions'] = to_numeric(df[adm_col]) if adm_col else df['fce']

    
    emg_col = find_col(cols, 'emergency')
    df['emergency'] = to_numeric(df[emg_col]) if emg_col else 0

    
    wl_col = find_col(cols, 'waiting list', 'waiting')
    df['waiting_list'] = to_numeric(df[wl_col]) if wl_col else 0

    
    los_col = find_col(cols, 'length of stay')
    # Prefer "mean" over "median" — find explicitly
    if los_col:
        mean_los_col = find_col([c for c in cols if 'mean' in c.lower()], 'length of stay')
        los_col = mean_los_col if mean_los_col else los_col
    df['mean_los_days'] = to_numeric(df[los_col]) if los_col else 0

    
    age_col = find_col(cols, 'mean age')
    df['mean_age'] = to_numeric(df[age_col]) if age_col else 0

    df['year'] = year

    
    df['diagnosis_desc'] = df['diagnosis_desc'].str.strip()

    
    # Keep codes that are a range (contain hyphen) or a single letter
    code = df['diagnosis_code'].astype(str).str.strip()
    is_range  = code.str.contains('-', na=False)
    is_single = code.str.match(r'^[A-Z]$', na=False)
    df = df[is_range | is_single].copy()

    return df[['diagnosis_code', 'diagnosis_desc', 'fce', 'admissions',
               'emergency', 'waiting_list', 'mean_los_days', 'mean_age', 'year']]



def collect_files():
    """Returns list of (filepath, year) tuples."""
    files = []

    # Old files: year subfolders 1998–2011
    for yr in range(1998, 2012):
        folder = os.path.join(BASE_DIR, str(yr))
        if not os.path.isdir(folder):
            print(f"WARNING: folder not found: {folder}")
            continue
        xls_files = glob.glob(os.path.join(folder, '*.xls'))
        # Pick the summary file: prefer filename containing 'sum';
        # fall back to the file that is neither 3cha nor 4cha
        summary = [f for f in xls_files if 'sum' in os.path.basename(f).lower()]
        if not summary:
            summary = [f for f in xls_files
                       if '3cha' not in os.path.basename(f).lower()
                       and '4cha' not in os.path.basename(f).lower()]
        if not summary:
            print(f"WARNING: no summary file in {folder}")
            continue
        files.append((summary[0], yr))

    # New files: root xlsx files 2012–2023
    xlsx_files = glob.glob(os.path.join(BASE_DIR, 'hosp-epis-stat-admi-diag-*.xlsx'))
    for fp in sorted(xlsx_files):
        name = os.path.basename(fp)
        # Extract the start year from e.g. "2023-24" → 2023
        m = re.search(r'diag-(\d{4})-\d{2}', name)
        if not m:
            print(f"WARNING: cannot parse year from {name}")
            continue
        yr = int(m.group(1))
        files.append((fp, yr))

    # Sort by year
    files.sort(key=lambda x: x[1])
    return files



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = collect_files()
    print(f"\nFound {len(files)} files to process:\n")
    for fp, yr in files:
        print(f"  {yr}  {os.path.basename(fp)}")

    all_frames = []
    for fp, yr in files:
        name = os.path.basename(fp)
        try:
            df = read_file(fp, yr)
            n = len(df)
            if n < 10:
                print(f"  WARNING: year {yr} has only {n} rows — {name}")
            elif n > 350:
                print(f"  WARNING: year {yr} has {n} rows (>350, unexpectedly large) — {name}")
            else:
                print(f"  OK  year={yr}  rows={n}  {name}")
            all_frames.append(df)
        except Exception as e:
            print(f"  FAILED: {name} — {e}")

    if not all_frames:
        print("ERROR: no data loaded. Exiting.")
        return

    merged = pd.concat(all_frames, ignore_index=True)
    print(f"\nMerged shape before metrics: {merged.shape}")

    
    baseline = (
        merged[merged['year'] == 2018]
        [['diagnosis_desc', 'admissions', 'waiting_list']]
        .rename(columns={'admissions': 'baseline_admissions',
                         'waiting_list': 'baseline_waiting'})
    )

    merged = merged.merge(baseline, on='diagnosis_desc', how='left')

    
    merged['pct_change_admissions'] = np.where(
        (merged['baseline_admissions'].isna()) | (merged['baseline_admissions'] == 0),
        np.nan,
        ((merged['admissions'] - merged['baseline_admissions']) / merged['baseline_admissions']) * 100
    )

    merged['pct_change_waiting'] = np.where(
        (merged['baseline_waiting'].isna()) | (merged['baseline_waiting'] == 0),
        np.nan,
        ((merged['waiting_list'] - merged['baseline_waiting']) / merged['baseline_waiting']) * 100
    )

    merged['backlog_trap'] = (
        merged['year'].isin([2022, 2023]) &
        (merged['pct_change_admissions'] < -5) &
        (merged['pct_change_waiting'] > 5)
    )

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")

    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    print(f"Total rows in CSV        : {len(merged)}")
    years_present = sorted(merged['year'].unique().tolist())
    print(f"Unique years present     : {years_present}")
    print(f"Unique diagnosis cats    : {merged['diagnosis_desc'].nunique()}")

    # Years with < 10 categories
    year_counts = merged.groupby('year').size()
    sparse = year_counts[year_counts < 10]
    if len(sparse):
        print(f"Years with <10 categories: {sparse.to_dict()}")
    else:
        print("Years with <10 categories: none")

    # Top 5 backlog_trap categories in 2023
    print("\nTop 5 backlog_trap=True categories in 2023:")
    bt = merged[(merged['backlog_trap']) & (merged['year'] == 2023)]
    bt_sorted = bt.sort_values('pct_change_admissions').head(5)
    if bt_sorted.empty:
        print("  (none)")
    else:
        for _, row in bt_sorted.iterrows():
            print(f"  {row['diagnosis_desc'][:50]:<50}  "
                  f"adm%={row['pct_change_admissions']:+.1f}  "
                  f"wait%={row['pct_change_waiting']:+.1f}")

    # Sample rows
    print("\nSample rows:")
    for sample_yr in [2018, 2020, 2023]:
        sample = merged[merged['year'] == sample_yr].head(3)
        if sample.empty:
            print(f"  year {sample_yr}: no data")
        else:
            print(f"\n  --- year {sample_yr} ---")
            print(sample[['diagnosis_code', 'diagnosis_desc', 'admissions',
                           'emergency', 'waiting_list', 'year']].to_string(index=False))

    print("\nDone.")


if __name__ == '__main__':
    main()
