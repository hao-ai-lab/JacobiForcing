import json
from transformers import AutoTokenizer

# Set your .jsonl file path and tokenizer name/path
#json_path = "/checkpoint/lhu/data/CLLM2_openthought/merged/4k_samples_sft_length_20k_filtered_output_merged_data_v1_8_18.jsonl"
jsonl_path = "/checkpoint/lhu/data/CLLM2_openthought/merged/40k_samples_merged_data_v2_8_27_opencodeinstruct_progressive_noise_cyclic_cap_idx_0p5.jsonl"

tokenizer_name = "/checkpoint/lhu/models/Qwen2.5-Coder-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

with open(jsonl_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i > 3:
            break

        data = json.loads(line)
        print(f"\ndata id: {data['data_id']}")
        
        if "complete_training_sequence_ids" in data:
            ids = data["complete_training_sequence_ids"]

            print(f"\ncomplete training sequence length: {len(ids)}")
            
            #decoded = tokenizer.decode(ids)
            #print(f"\ndecoded length: {len(decoded)}")

            print(f"\n[Line {i}] clean tokens:\n\n{ids[-16:]}")
            print(f"\n[Line {i}] clean decoded:\n\n{tokenizer.decode(ids[-16:])}")

            print(f"\n[Line {i}] noisy tokens:\n\n{ids[-32:-16]}")
            print(f"\n[Line {i}] noisy decoded:\n\n{tokenizer.decode(ids[-32:-16])}")

        elif "labels_ids" in data:

            ids = data["labels_ids"]
            print(f"\nlabels length: {len(ids)}")
            decoded = tokenizer.decode(ids)
            print(f"\n[Line {i}] Decoded:\n\n{decoded}")

        #print(f"\ncomplete training sequence length: {len(data['complete_training_sequence_ids'])}")
        #print(f"\nlabel length: {len(data['labels_ids'])}")
        #print(f"prompt id: {data['prompt_ids']}")
        #print(f"\nprompt id length: {data['prompt_ids_len'][0]}")
