import json

import prompting

BASES_DIR = ""
CACHE_DIR = ""
STIMULI_DIR = ""


def load_bases(suffix):
    with open(BASES_DIR + f"/{suffix}.txt", "r") as f:
        bases = f.read().strip().split("\n")
    return [b.strip() for b in bases]


def load_stimuli(affixes):
    with open(STIMULI_DIR + f"/{affixes}.json", "r") as f:
        stimuli = json.load(f)
    return list(stimuli.keys())


def load_prompts(affixes):
    return prompting.AFFIXES2PROMPTS[affixes]


def generate_derivatives(base, suffix):
    if suffix == "ive" or suffix == "ive_nonce":
        derivative_ness = base + "ness"
        derivative_ity = base[:-1] + "ity"
        return derivative_ness, derivative_ity


def get_base_derivatives(stimulus, affixes):
    derivatives = stimulus.split("_")
    if affixes.startswith("iveness_ivity"):
        base = derivatives[0][:-4]
    elif affixes.startswith("ity_ness"):
        base = derivatives[1][:-4]
    return base, derivatives


def get_input_prompt(base, derivative, prompt, prompt_type):
    if prompt_type == "target":
        input_prompt = prompt + " " + derivative
    else:
        input_prompt = prompt.format(base) + " " + derivative
    return input_prompt
