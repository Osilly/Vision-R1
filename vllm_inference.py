import base64
from openai import OpenAI
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Vision-R1 model inference.")
    parser.add_argument("--model_path", type=str, default="", help="Path to the model.")
    parser.add_argument("--image_path", type=str, default="", help="Path to the input image.")
    parser.add_argument("--prompt", type=str, default="", help="The input prompt.")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens of model generation")
    parser.add_argument("--temperature", type=float, default=0.01, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.1, help="top_p")
    args = parser.parse_args()

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    image_path = args.image_path
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_r1 = f"data:image;base64,{encoded_image_text}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": base64_r1},
                },
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    chat_response = client.chat.completions.create(
        model=args.model_path,
        messages=messages,
        stream=False,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
    )
    # chat_response = chat_response.choices[0].message.content
    print("Chat response:", chat_response)

if __name__ == "__main__":
    main()
