# Inference using vLLMs
python vllm_inference.py \\
--model_path  ""  \\
--image_path "./figs/example1.png" \\
--prompt "Given a cone with a base radius represented by the variable 'r' (r = 1) and a slant height represented by the variable 's' (s = 3), determine the lateral surface area using variables.\nChoices:\nA: 2π\nB: 3π\nC: 6π\nD: 8π" \\
--max_tokens 2048 \\
--temperature 0.01 \\
--top_p 0.1 \\
--timeout 2000 
