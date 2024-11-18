import argparse
import json
import tqdm

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)

import helpers


def main():

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--affixes",
        type=str,
        required=True,
        help="Affixes to examine."
    )
    args = parser.parse_args()

    # Load tokenizer and model
    model_name = "EleutherAI/gpt-j-6B"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=helpers.CACHE_DIR,
        device_map="auto"
    )

    # Load prompts
    prompts = helpers.load_prompts(args.affixes)

    # Load rivals
    stimuli = helpers.load_stimuli(args.affixes)

    # Create dictionary to store results
    results_dict = {}

    # Initialize log softmax
    log_softmax = torch.nn.LogSoftmax(dim=1)

    # Evaluate derivatives
    for prompt_type in prompts:
        results_dict[prompt_type] = {}
        for prompt in prompts[prompt_type]:
            print(f'''Processing prompt "{prompt}"...''')
            results_dict[prompt_type][prompt] = {}
            for stimulus in tqdm.tqdm(stimuli):

                # Initialize entry for stimulus
                results_dict[prompt_type][prompt][stimulus] = {}

                # Get base and derivatives
                base, derivatives = helpers.get_base_derivatives(
                    stimulus, 
                    args.affixes
                )
            
                # Loop over derivatives
                for derivative in derivatives:

                    # Create tokenization of derivative
                    tok_derivative = tok.tokenize(" " + derivative)

                    # Create input prompt
                    input_prompt = helpers.get_input_prompt(
                        base, 
                        derivative, 
                        prompt, 
                        prompt_type
                    )

                    # Pass input prompt through model
                    input_ids = tok(
                        input_prompt, 
                        return_tensors="pt"
                    ).input_ids.to("cuda")
                    output = model(input_ids, labels=input_ids)

                    # Compute log probabilities of tokens
                    logits = output[1][..., :-1, :].contiguous()
                    logits = logits.view(-1, logits.size(-1))
                    logprobs = log_softmax(logits)
                    labels = input_ids[..., 1:].contiguous()
                    labels = labels.view(-1)
                    logprobs_labels = logprobs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
                    logprobs_derivative = logprobs_labels[-len(tok_derivative):].tolist()

                    # Store results
                    results_dict[prompt_type][prompt][stimulus][derivative] = [
                        tok_derivative, logprobs_derivative
                    ]
                
    with open(f"gptj_{args.affixes}.json", "w") as f:
        json.dump(results_dict, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
