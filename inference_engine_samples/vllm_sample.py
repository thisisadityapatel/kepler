from vllm import LLM, SamplingParams


def main():
    print("Loading model...")

    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
        max_model_len=1000,
        gpu_memory_utilization=0.5,
    )

    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=100)

    prompts = [
        "What is the capital of India?",
        "Write a haiku about programming.",
        "Explain what vLLM is in one sentence.",
    ]

    print("\nGenerating responses...\n")

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {i + 1}: {prompt}")
        print(f"Response: {generated_text}")
        print("—" * 20)


if __name__ == "__main__":
    main()
