import json
from tqdm import tqdm
import os


def merge_jsonl_files(input_files, output_file):
    """
    Merge multiple JSONL files into a single JSONL file with a global progress bar.

    Args:
        input_files (list of str): List of input file paths.
        output_file (str): Path to the merged output file.
    """
    # Count total lines across all files
    total_lines = 0
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as f:
            total_lines += sum(1 for _ in f)

    with open(output_file, 'w', encoding='utf-8') as outfile, tqdm(total=total_lines, desc="Merging JSONL files") as pbar:
        for file in input_files:
            with open(file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:  # skip empty lines
                        try:
                            json_obj = json.loads(line)
                            outfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                        except json.JSONDecodeError as e:
                            print(f"Skipping invalid JSON in {file}: {e}")
                    pbar.update(1)

if __name__ == "__main__":
    # Example usage
    input_files = [
        "/checkpoint/lhu/data/CLLM2_data_prep/trajectory_bs_k8s_08_27_merged/merged_all_08_27_lanxiang.jsonl",
        "/checkpoint/lhu/data/TRACE-08-27/merged_all_08_27_yichao.jsonl",
        "/checkpoint/lhu/data/opencoderinstruct_trajectory/merged/merged_all_8_27_siqi.jsonl"
    ]
    output_file = "/checkpoint/lhu/data/CLLM2_openthought/merged/merged_data_v2_8_28_raw.jsonl"
    merge_jsonl_files(input_files, output_file)
    print(f"Merged {len(input_files)} files into {output_file}")
