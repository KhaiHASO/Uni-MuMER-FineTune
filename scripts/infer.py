# -*- coding: utf-8 -*-
"""
Inference script with 4-bit quantization.
Uses BitsAndBytes 4-bit quantization for memory-efficient inference.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import editdistance
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)

# Optional: official Qwen visual pre-processing
try:
    from qwen_vl_utils import (
        process_vision_info,
        vision_process,
    )
except ModuleNotFoundError:
    vision_process = None

from eval_metrics_calculator import evaluate_text_generation

# -----------------------------------------------------------------------------#
# Logging
# -----------------------------------------------------------------------------#
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

def compute_distance(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between *a* and *b*."""
    return editdistance.eval(a.split(), b.split())

def quick_report(results: List[Dict[str, str]]) -> None:
    """Print *exprate*, *error1*, and *error2* on three separate lines."""
    total = len(results)
    if total == 0:
        for tag in ("exprate", "error1", "error2"):
            print(f"{tag}: 0.00%")
        return

    dists = [compute_distance(r["pred"], r["gt"]) for r in results]

    def percentage(count: int) -> str:
        return f"{count / total * 100:.2f}%"

    exprate = percentage(sum(d == 0 for d in dists))
    error1  = percentage(sum(d <= 1 for d in dists))
    error2  = percentage(sum(d <= 2 for d in dists))

    print(f"exprate: {exprate}")
    print(f"error1:  {error1}")
    print(f"error2:  {error2}")

def run_inference(
    model_name: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    suffix: str = "_pred",
    max_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.8,
    top_k: int = 50,
    batch_size: int = 1,
    use_4bit: bool = True,
    quant_type: str = "nf4",
) -> None:
    """
    Run inference with optional 4-bit quantization for memory efficiency.
    Processes full datasets (no sample limiting).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 60)
    LOGGER.info("PRODUCTION INFERENCE - 4-bit Quantization")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Batch size: {batch_size}")
    LOGGER.info(f"Max tokens: {max_tokens}")
    LOGGER.info(f"4-bit quantization: {use_4bit}")
    if use_4bit:
        LOGGER.info(f"Quantization type: {quant_type}")
    LOGGER.info("=" * 60)

    # Clear cache before loading
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        LOGGER.info(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

    # 1) Load model with optional 4-bit quantization -----------------------#
    LOGGER.info("Loading model...")
    
    if use_4bit:
        # Cấu hình nén 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=quant_type,  # "nf4" or "fp4"
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        LOGGER.info(f"Loading model with 4-bit {quant_type} quantization...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Tự động đẩy vào GPU
            trust_remote_code=True,
        )
    else:
        # Load without quantization (uses more memory)
        LOGGER.info("Loading model without quantization...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    model.eval()  # Set to evaluation mode
    
    if torch.cuda.is_available():
        LOGGER.info(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

    # 2) File loop ----------------------------------------------------------#
    for json_path in sorted(input_dir.glob("*.json")):
        LOGGER.info("[FILE] %s", json_path.name)
        with json_path.open(encoding="utf-8") as fp:
            dataset: List[Dict] = json.load(fp)

        results: List[Dict] = []
        reqs_batch: List[Dict] = []
        metas_batch: List[Dict] = []

        def flush_batch():
            """Process current batch and collect outputs."""
            if not reqs_batch:
                return
            
            LOGGER.info(f"Processing batch of {len(reqs_batch)} samples...")
            batch_results = []
            
            for inputs_dict, meta in zip(reqs_batch, metas_batch):
                # Generate for single sample
                with torch.no_grad():
                    generate_kwargs = {
                        "max_new_tokens": max_tokens,
                        "pad_token_id": processor.tokenizer.eos_token_id,
                    }
                    
                    if temperature > 0:
                        generate_kwargs.update({
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                            "do_sample": True,
                        })
                    else:
                        generate_kwargs["do_sample"] = False
                    
                    generated_ids = model.generate(**inputs_dict, **generate_kwargs)
                
                # Decode - extract only the generated part
                input_length = inputs_dict["input_ids"].shape[1]
                generated_ids_trimmed = generated_ids[0][input_length:]
                output_text = processor.tokenizer.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                ).strip()
                
                batch_results.append({
                    "gt": meta["gt"],
                    "pred": output_text,
                    "image_path": meta["image_path"],
                    "img_id": Path(meta["image_path"]).stem,
                })
            
            results.extend(batch_results)
            reqs_batch.clear()
            metas_batch.clear()
            torch.cuda.empty_cache()  # Clear cache after each batch

        valid_cnt = 0
        for record in tqdm(dataset, desc="Processing records", unit="record"):
            if not record.get("images"):
                continue
            image_path = record["images"][0]

            prompt_text = gt_text = None
            for msg in record.get("messages", []):
                if msg["from"] == "human":
                    prompt_text = msg["value"].strip()
                elif msg["from"] == "gpt":
                    gt_text = msg["value"].strip()
            if not prompt_text or gt_text is None:
                continue

            # Prepare messages for Qwen2.5-VL (same format as vllm_infer.py)
            image_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
            
            # Apply chat template to get text prompt
            text = processor.apply_chat_template(
                image_messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info using qwen_vl_utils if available
            if vision_process is not None:
                try:
                    image_inputs, _, _ = process_vision_info(image_messages, return_video_kwargs=True)
                    images = image_inputs if image_inputs is not None else []
                except Exception as e:
                    LOGGER.warning(f"Error processing vision info: {e}, falling back to direct image load")
                    from PIL import Image
                    image = Image.open(image_path).convert("RGB")
                    images = [image]
            else:
                # Fallback: load image directly
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
                images = [image]
            
            # Process with processor (handles both text and images)
            inputs = processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt"
            )
            
            # Move all inputs to model device
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}

            reqs_batch.append(inputs)
            metas_batch.append({"gt": gt_text, "image_path": image_path})
            valid_cnt += 1

            # Flush batch when it reaches batch_size
            if len(reqs_batch) >= batch_size:
                flush_batch()

        # Process remaining batch
        flush_batch()

        if not results:
            LOGGER.info("↳ No valid sample in %s – skipped.", json_path.name)
            continue

        LOGGER.info("↳ Generated %d records from %s", len(results), json_path.name)

        # 3) Save results -----------------------------------------------------#
        out_file = output_dir / f"{json_path.stem}{suffix}.json"
        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("↳ Saved %d records → %s", len(results), out_file)

        # 4) Evaluate results -------------------------------------------------#
        metrics = evaluate_text_generation(
            out_file,
            os.path.join(output_dir, f"{json_path.stem}_results.txt")
        )

    # Cleanup
    del model
    torch.cuda.empty_cache()
    LOGGER.info("Inference completed. Model unloaded from memory.")

# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference with 4-bit quantization"
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model path or Hugging Face repo.",
    )
    parser.add_argument(
        "--input-dir",
        default="./data/",
        help="Directory containing source JSON files. (default: ./data/)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write prediction JSON files. (default: outputs)",
    )
    parser.add_argument(
        "--suffix",
        default="_pred",
        help='Suffix appended to output files (default: "_pred").',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate (default: 2048).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling top-p (default: 0.8).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1, increase if you have more VRAM).",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (uses more memory, not recommended).",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="Quantization type: nf4 (default) or fp4.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("INFERENCE - 4-bit Quantization")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"4-bit quantization: {not args.no_4bit}")
    if not args.no_4bit:
        print(f"Quantization type: {args.quant_type}")
    print("=" * 60)
    
    run_inference(
        model_name=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        suffix=args.suffix,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=50,
        batch_size=args.batch_size,
        use_4bit=not args.no_4bit,
        quant_type=args.quant_type,
    )

if __name__ == "__main__":
    main()

