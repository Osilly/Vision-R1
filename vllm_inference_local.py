from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Vision-R1 model inference.")
    parser.add_argument("--model_path", type=str, default="", help="Path to the model.")
    parser.add_argument("--image_path", type=str, default="", help="Path to the input image.")
    parser.add_argument("--prompt", type=str, default="", help="The input prompt.")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens of model generation")
    parser.add_argument("--temperature", type=float, default=0.01, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.1, help="top_p")
    parser.add_argument("--timeout", type=int, default=2000, help="timeout")
    args = parser.parse_args()
    

    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 5},
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    image_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_path,
                },
                {
                    "type": "text", 
                    "text": args.prompt},
            ],
        },
    ]

    # Here we use video messages as a demonstration
    messages = image_messages

    processor = AutoProcessor.from_pretrained(args.model_path)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)

    mm_data = {}
    mm_data["image"] = image_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    print(generated_text)

if __name__ == "__main__":
    main()