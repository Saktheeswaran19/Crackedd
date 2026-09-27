"""
Helper script to merge India and US candidate pairs into a single train_candidate_pairs.tsv file.

Run from student_resource/ root:
    python code\\business_entity_resolution\\src\\merge_candidates.py
"""

import os
import sys

SRC_DIR = os.path.dirname(__file__)
INDIA_FILE = os.path.join(SRC_DIR, "train_candidate_pairs_India.tsv")
US_FILE = os.path.join(SRC_DIR, "train_candidate_pairs_US.tsv")
MERGED_FILE = os.path.join(SRC_DIR, "train_candidate_pairs.tsv")

def main():
    print("Checking input candidate files...")
    if not os.path.exists(INDIA_FILE):
        print(f"Error: {INDIA_FILE} does not exist. Please run India blocking first.")
        sys.exit(1)
    if not os.path.exists(US_FILE):
        print(f"Error: {US_FILE} does not exist. Please run US blocking first.")
        sys.exit(1)

    print(f"Merging {INDIA_FILE} and {US_FILE} into {MERGED_FILE}...")
    total_lines = 0

    with open(MERGED_FILE, 'w', encoding='utf-8') as out_f:
        # Write India file with header
        with open(INDIA_FILE, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                out_f.write(line)
                total_lines += 1

        # Write US file skipping header
        with open(US_FILE, 'r', encoding='utf-8') as us_f:
            header = us_f.readline()  # skip header
            for line in us_f:
                out_f.write(line)
                total_lines += 1

    print(f"Successfully merged candidate pairs: {total_lines - 1:,} records written to {MERGED_FILE}")

if __name__ == '__main__':
    main()
