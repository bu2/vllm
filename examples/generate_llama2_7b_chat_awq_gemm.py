import time

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "[INST]How does `A → B` relate to `¬A ∨ B`?[/INST]",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=512, temperature=0, top_p=0.95, seed=0)

# Create an LLM.
llm = LLM(model="../AutoAWQ/llama-2-7b-chat-hf-awq-gemm")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
tstart = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
tend = time.perf_counter()
print("Generation: %fs elapsed." % (tend-tstart))
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\n\n")
    print(generated_text)
