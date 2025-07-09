import json
import argparse
import statistics

def smart_open(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f]

def describe_list(lst):
    if not lst:
        return "[]"
    if all(isinstance(x, int) for x in lst):
        return f"[int] len={len(lst)}"
    if all(isinstance(x, list) and all(isinstance(xx, int) for xx in x) for x in lst):
        lens = [len(x) for x in lst]
        return f"[[int,…],…] {len(lst)} chunks (chunk_lens={lens})"
    return f"{type(lst).__name__} of {len(lst)}"

def profile(file, max_rows=100):
    data = smart_open(file)
    n = min(max_rows, len(data))
    print(f"Profiling {n} / {len(data)} rows from: {file}\n")

    # Gather per-field values
    fields = [
        "data_id", 
        "diffusion_itr_id", 
        "prompt_ids_len",
        "prompt_ids", 
        "answer_trajectory_ids",
        "teacher_output_ids", 
        "labels_ids"
    ]
    stats = {k: [] for k in fields}
    for row in data[:n]:
        for k in fields:
            stats[k].append(row.get(k))

    # Print the schema-aligned summary
    def ex(x):  # get an example value
        for v in x:
            if v is not None:
                return v

    for k in fields:
        v = stats[k]
        sample = ex(v)
        # Types
        if k in ("data_id", "diffusion_itr_id"):
            print(f'{k:<22} str\t\te.g. "{sample}"')
        elif k == "prompt_ids_len":
            print(f'{k:<22} [int]\t\te.g. {sample}')
        elif k == "prompt_ids":
            lens = [len(x) for x in v if isinstance(x, list)]
            print(f'{k:<22} [int,…]\tlen={statistics.mean(lens):.1f} (min={min(lens)}, max={max(lens)})')
        elif k == "answer_trajectory_ids":
            if isinstance(sample, list) and sample and isinstance(sample[0], list):
                num_chunks = [len(x) for x in v if isinstance(x, list)]
                flat_lens = [[len(xx) for xx in x] for x in v if isinstance(x, list)]
                print(f'{k:<22} [[int,…],…]\tnum_chunks={statistics.mean(num_chunks):.1f} per row; chunk_lens (first row)={flat_lens[0] if flat_lens else "[]"}')
            else:
                print(f'{k:<22} [[int,…],…]\tempty or missing')
        elif k in ("teacher_output_ids", "labels_ids"):
            lens = [len(x) for x in v if isinstance(x, list)]
            print(f'{k:<22} [int,…]\tlen={statistics.mean(lens):.1f} (min={min(lens)}, max={max(lens)})')
    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--max_rows", type=int, default=100000)
    args = parser.parse_args()
    profile(args.file, args.max_rows)
