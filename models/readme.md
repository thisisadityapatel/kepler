Download model .gguf weights from hugging face using the `huggingface-cli` library.

### Example

Run the following command to download the .gguf into the current directory:

```shell
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q5_k_m.gguf --local-dir .
```

NOTE: the .gguf file needs to be in the `models` directory to be picked up by Kepler.