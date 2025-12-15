import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True, help="Output for kept entries.")
    parser.add_argument("--max-seq-len", type=int, required=True)
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as fin, \
         open(args.output_path, "w", encoding="utf-8") as fout_keep:
        for line in fin:
            item = json.loads(line)
            seq = item.get("complete_training_sequence_ids", [])
            if len(seq) <= args.max_seq_len:
                print(f"Keeping item {item['data_id']} with sequence length {len(seq)}")
                fout_keep.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
