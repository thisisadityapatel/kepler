from llama_cpp import Llama


def main():
    llm = Llama(
        model_path="../models/qwen2.5-0.5b-instruct-q5_k_m.gguf",
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=0,
        verbose=True,
    )

    prompts = [
        "What is the capital of India?",
        "Write a haiku about programming.",
        "Explain what vLLM is in one sentence.",
    ]

    for i, prompt in enumerate(prompts):
        print(f"Prompt {i + 1}: {prompt}")

        output = llm(
            prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["User:", "\n\n"],
        )

        response = output["choices"][0]["text"]
        print(f"Response: {response}")
        print("-" * 80)


if __name__ == "__main__":
    main()
