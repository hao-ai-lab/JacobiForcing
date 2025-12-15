import os
import json
import argparse
import pyarrow.parquet as pq

def filter_and_rank(input_dir: str, output_path: str):
    """
    Filters OpenCodeInstruct data entries with average_test_score == 1.0,
    ranks them by llm_judgement['score'], and writes sorted entries to a JSONL file.
    """
    records = []

    parquet_files = sorted(
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".parquet")
    )

    for pfile in parquet_files:
        print(f"Reading {pfile}...")
        table = pq.read_table(pfile)
        df = table.to_pandas()

        perfect = df[df['average_test_score'].astype(float) == 1.0]
        print(f" --> Found {len(perfect)} perfect-score records in this file.")

        for _, row in perfect.iterrows():
            lj = json.loads(row['llm_judgement'])
            score = float(lj.get("score", 0))

            rec = row.to_dict()
            rec['llm_score'] = score
            records.append(rec)

    print(f"Sorting {len(records)} records by LLM judgement score (descending)...")
    records.sort(key=lambda x: x['llm_score'], reverse=True)

    print(f"Writing sorted records to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for rec in records:
            rec.pop('llm_score', None)
            f_out.write(json.dumps(rec) + "\n")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter & rank OpenCodeInstruct entries by llm_judgement score"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing .parquet input files"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to output .jsonl file"
    )
    args = parser.parse_args()

    filter_and_rank(args.input_dir, args.output_path)
