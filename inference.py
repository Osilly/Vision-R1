from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Run Vision-R1 model inference.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the model.")
    parser.add_argument("--enable_flash_attn", type=bool, default=True, help="Enable flash-attention for better acceleration and memory saving.")
    parser.add_argument("--image_path", type=str, default="", help="Path to the input image.")
    parser.add_argument("--prompt", type=str, default="", help="The input prompt.")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens of model generation")
    args = parser.parse_args()

    if args.enable_flash_attn:
        # need to install flash-attention first.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto"
        )

    # default processor
    processor = AutoProcessor.from_pretrained(args.model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_path,
                },
                {
                    "type": "text", 
                    "text": args.prompt,
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

if __name__ == "__main__":
    main()