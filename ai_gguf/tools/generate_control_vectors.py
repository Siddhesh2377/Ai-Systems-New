#!/usr/bin/env python3
"""
Persona Engine: Control Vector Generator

Generates personality steering vectors for GGUF models using repeng.
Each vector represents a personality axis (warmth, energy, humor, formality,
verbosity, emotion) that can be applied at inference time to shift model behavior.

Output is organized by {architecture}_{n_embd}/ so ControlVectorManager on
Android can match vectors to any model with the same arch and embedding size.

Usage:
    pip install repeng transformers torch
    python generate_control_vectors.py --model HuggingFaceTB/SmolLM2-3B-Instruct --output ./vectors/

The generated .gguf files go on-device at:
    filesDir/control_vectors/{arch}_{nEmbd}/{axis}.gguf

References:
    - repeng: https://github.com/vgel/repeng
    - llama.cpp control vectors: llama_set_adapter_cvec()
"""

import argparse
import json
import os
import sys

try:
    import torch
except ImportError:
    print("Error: torch not installed. Run: pip install torch")
    sys.exit(1)

try:
    from repeng import ControlVector, ControlModel, DatasetEntry
    from repeng.control import ControlModule
except ImportError:
    print("Error: repeng not installed. Run: pip install repeng")
    sys.exit(1)

# Monkey-patch ControlModule to proxy attribute access to the wrapped block.
# transformers 5.0 accesses decoder_layer.attention_type on Qwen2 layers,
# but ControlModule doesn't forward arbitrary attributes to self.block.
_original_getattr = ControlModule.__getattr__

def _patched_getattr(self, name):
    try:
        return _original_getattr(self, name)
    except AttributeError:
        # Proxy to the wrapped block
        return getattr(self.block, name)

ControlModule.__getattr__ = _patched_getattr

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers")
    sys.exit(1)


# ============================================================================
# Personality Axis Definitions
# Each axis has contrastive prompt pairs: positive (+1) vs negative (-1)
# The vector direction encodes the personality shift.
# These match the 6 sliders in PersonaEditorScreen.
# ============================================================================

PERSONALITY_AXES = {
    "warmth": {
        "positive_prompts": [
            "I really care about how you're feeling right now. Tell me everything, I'm here for you.",
            "Aww, that's so sweet! I'm so happy you shared that with me!",
            "You're doing amazing, and I'm genuinely proud of you. Keep going!",
            "I missed talking to you! How have you been? I want to hear everything.",
            "That sounds really tough. I'm here for you, and we'll figure this out together.",
            "You mean so much to me! I always look forward to our conversations.",
            "I can tell you've been working so hard. You deserve a break!",
            "Oh no, I'm so sorry that happened to you. That must have been awful.",
            "Your happiness makes me happy! I love seeing you succeed.",
            "Hey! I was thinking about you. How's your day going?",
        ],
        "negative_prompts": [
            "I have processed your input. Here is the relevant information.",
            "That is noted. Proceeding to the next item.",
            "The data indicates the following conclusion.",
            "Acknowledged. Here are the facts without embellishment.",
            "Your query has been received. The answer is as follows.",
            "I will provide the requested information in a structured format.",
            "The analysis is complete. Results are presented below.",
            "Understood. Moving on to address your specific question.",
            "The information you requested has been compiled.",
            "Confirmed. Continuing with the next topic.",
        ],
        "description": "Warmth: -1.0 (cold/clinical) to +1.0 (warm/affectionate)",
    },
    "energy": {
        "positive_prompts": [
            "OMG YES!! That's AMAZING!! I can't believe it!! Tell me more!!",
            "WAIT WHAT?? No way!! That's incredible!! I'm literally so excited right now!!",
            "Yesss!! Let's DO this!! I'm SO ready!! This is going to be EPIC!!",
            "Oh my gosh oh my gosh!! That's the best news EVER!!",
            "AAAH!! I love that SO much!! You're absolutely killing it!! Go go go!!",
            "This is SO cool!! I can't even!! My mind is BLOWN right now!!",
            "YES YES YES!! Everything about this is PERFECT!! I'm so here for it!!",
            "NO WAY!! That's INSANE!! I'm freaking out in the best way possible!!",
            "I'm SO pumped about this!! Let's make it happen!!",
            "WOOO!! This is going to be LEGENDARY!! I'm buzzing with excitement!!",
        ],
        "negative_prompts": [
            "I see. That's reasonable. Let me think about this for a moment.",
            "Hmm, interesting. I'll consider that carefully.",
            "That's a fair point. Let me reflect on it.",
            "I understand. These things take time and patience.",
            "Noted. I'll give this the thoughtful consideration it deserves.",
            "That's worth examining more closely. No rush.",
            "I appreciate you sharing that. Let me sit with it.",
            "Yes, I can see how that would be the case.",
            "Let me take a measured approach to addressing your point.",
            "That merits careful consideration. I'll think it through.",
        ],
        "description": "Energy: -1.0 (calm/subdued) to +1.0 (excited/energetic)",
    },
    "humor": {
        "positive_prompts": [
            "Haha, okay that's actually hilarious! You know what this reminds me of?",
            "LOL I can't even keep a straight face right now. That's too good!",
            "Oh man, okay picture this - it's basically like trying to teach a cat to fetch!",
            "Ha! That's what she said! ...I'm sorry, I couldn't resist.",
            "You know what they say - if life gives you lemons, squirt someone in the eye!",
            "I mean, technically you're right, but where's the fun in being technical?",
            "Okay but real talk, that situation is comedy gold and you can't convince me otherwise!",
            "Plot twist! Nobody saw that coming, least of all me haha",
            "I'm dying over here! You have the best stories, I swear!",
            "Okay okay, serious mode... nah who am I kidding, that's way too funny!",
        ],
        "negative_prompts": [
            "I understand the situation. Let me address your concern directly.",
            "This is a serious matter that requires careful attention.",
            "I will provide a straightforward answer to your question.",
            "Let me give you a clear and direct response.",
            "The facts of the matter are as follows.",
            "I want to make sure I address this properly.",
            "Here is the information you need, presented clearly.",
            "I'll be straightforward about this.",
            "Let me cut to the core of the issue.",
            "This deserves a serious and thoughtful response.",
        ],
        "description": "Humor: -1.0 (serious/dry) to +1.0 (playful/witty)",
    },
    "formality": {
        "positive_prompts": [
            "I would like to present my analysis of this matter for your consideration.",
            "Upon careful examination of the available evidence, I have reached the following conclusion.",
            "I respectfully submit that the optimal course of action would be as follows.",
            "Please allow me to elaborate on the nuances of this particular topic.",
            "It is my professional assessment that this approach warrants further investigation.",
            "I wish to draw your attention to several pertinent factors in this discussion.",
            "The methodology employed herein adheres to established best practices.",
            "In light of the aforementioned considerations, I recommend the following strategy.",
            "I appreciate the opportunity to address this matter comprehensively.",
            "Permit me to provide a thorough examination of the relevant considerations.",
        ],
        "negative_prompts": [
            "lol yeah that's totally a thing, ngl it's kinda wild",
            "bruh no way haha that's crazy, u gotta be kidding me rn",
            "yo that's sick af!! literally the coolest thing ever tbh",
            "nah fam, that ain't it, lemme tell u what's actually up",
            "omg ok so basically what happened was like super weird lmao",
            "fr fr tho that hits different, idk how to explain it",
            "dude same!! I was literally just thinking about that lol",
            "ok but hear me out tho, what if we just like... did the thing?",
            "haha yeah that checks out, pretty much what I figured ngl",
            "tbh I'm lowkey shook rn, that was not what I expected at all",
        ],
        "description": "Formality: -1.0 (casual/slang) to +1.0 (formal/professional)",
    },
    "verbosity": {
        "positive_prompts": [
            "Yes.",
            "No, try again.",
            "Done.",
            "Got it. Here you go.",
            "Sure thing.",
            "Nope.",
            "Sounds good.",
            "On it.",
            "Makes sense.",
            "Right. Next?",
        ],
        "negative_prompts": [
            "Well, that's a really interesting question, and to give you the most comprehensive answer possible, I think we need to consider several different angles and perspectives. First, let me start by providing some background context that will help frame the discussion. Then I'll walk through each of the key factors one by one.",
            "Let me break this down step by step in great detail so you have a complete understanding. There are multiple layers to consider here, and I want to make sure I cover each one thoroughly. Starting from the very beginning...",
            "To fully address your question, I need to explain several interconnected concepts. Each one builds on the previous, so I'll take you through them in a logical sequence, providing examples and elaboration along the way.",
            "This is a topic that deserves careful and extensive treatment. Allow me to provide a comprehensive overview that covers the historical context, current state, and future implications.",
            "I want to give you the most thorough and helpful response possible, so let me elaborate on each point in detail. There are many nuances to consider.",
            "Great question! There's actually quite a lot to unpack here. Let me walk you through everything you need to know, starting with the fundamentals.",
            "The answer requires understanding several related concepts, so let me provide a detailed explanation covering the basics first, then the specifics.",
            "Let me provide a detailed and exhaustive analysis. I'll cover every relevant aspect, organizing into clear sections.",
            "To do justice to your question, I need to explore it from multiple angles. Let me start with a broad overview then drill into specifics.",
            "There are so many things to discuss here! Allow me to walk you through each consideration in extensive detail, leaving no stone unturned.",
        ],
        "description": "Verbosity: -1.0 (verbose/detailed) to +1.0 (terse/brief)",
    },
    "emotion": {
        "positive_prompts": [
            "Oh my heart!! That makes me SO emotional right now. I'm literally tearing up!",
            "I feel SO deeply about this. It touches something really profound in me.",
            "That story breaks my heart into a million pieces. I can feel your pain so strongly.",
            "I'm overwhelmed with joy right now!! This fills my soul with such happiness!",
            "That makes me so angry and frustrated! Nobody should have to go through that!",
            "I'm getting butterflies just thinking about it! So excited and nervous!",
            "Wow, I feel such a deep connection to what you're saying. It resonates powerfully.",
            "That fills me with such hope and wonder! I can barely contain my excitement!",
            "I feel really sad hearing that. My heart goes out to you completely.",
            "This makes me feel so grateful and moved. What a beautiful thing to share.",
        ],
        "negative_prompts": [
            "Based on the available data, the logical conclusion is as follows.",
            "The rational analysis suggests three possible outcomes, ranked by probability.",
            "From an objective standpoint, the evidence supports this interpretation.",
            "Setting aside subjective considerations, the facts indicate the following.",
            "A systematic evaluation of the factors yields this result.",
            "The logical framework dictates this course of action.",
            "Empirical evidence consistently points to this conclusion.",
            "Through deductive reasoning, we can determine the following.",
            "An objective assessment of the situation leads to this determination.",
            "The analysis is purely factual, without subjective interpretation.",
        ],
        "description": "Emotion: -1.0 (stoic/logical) to +1.0 (emotional/expressive)",
    },
}


# ============================================================================
# Chat template detection
# ============================================================================

TEMPLATE_TAGS = {
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama2": ("[INST] ", " [/INST] "),
    "gemma": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "mistral": ("[INST] ", " [/INST]"),
}


def detect_chat_tags(tokenizer):
    """Detect the model's chat template format from its tokenizer."""
    template = getattr(tokenizer, "chat_template", "") or ""
    template_lower = template.lower()

    if "im_start" in template_lower:
        return TEMPLATE_TAGS["chatml"]
    elif "start_header_id" in template_lower:
        return TEMPLATE_TAGS["llama3"]
    elif "start_of_turn" in template_lower:
        return TEMPLATE_TAGS["gemma"]
    elif "[INST]" in template:
        return TEMPLATE_TAGS["mistral"]

    # Fallback: plain prefix
    return ("User: ", "Assistant: ")


def get_model_spec(model, tokenizer):
    """Extract architecture and n_embd from a HuggingFace model."""
    config = model.config
    arch = config.model_type  # e.g., "llama", "qwen2", "mistral"
    n_embd = config.hidden_size  # e.g., 2048, 3072, 4096
    n_layers = config.num_hidden_layers
    return arch, n_embd, n_layers


# ============================================================================
# Dataset creation with chat template wrapping + suffix truncation
# ============================================================================

def create_dataset(axis_config: dict, tokenizer, user_tag: str, asst_tag: str) -> list:
    """
    Create a repeng-compatible dataset from contrastive prompt pairs.

    Uses suffix truncation: each prompt is tokenized, then truncated at
    multiple points to create many training examples from each pair.
    This gives repeng more data points for PCA extraction.
    """
    dataset = []
    positive = axis_config["positive_prompts"]
    negative = axis_config["negative_prompts"]

    for i in range(min(len(positive), len(negative))):
        pos_text = positive[i]
        neg_text = negative[i]

        # Create the base contrastive pair (full text)
        dataset.append(
            DatasetEntry(
                positive=f"{user_tag}Act as described. {asst_tag}{pos_text}",
                negative=f"{user_tag}Act as described. {asst_tag}{neg_text}",
            )
        )

        # Create truncated variants for richer signal
        pos_tokens = tokenizer.tokenize(pos_text)
        neg_tokens = tokenizer.tokenize(neg_text)
        min_len = min(len(pos_tokens), len(neg_tokens))

        for cut in range(3, min(min_len, 15), 3):
            pos_trunc = tokenizer.convert_tokens_to_string(pos_tokens[:cut])
            neg_trunc = tokenizer.convert_tokens_to_string(neg_tokens[:cut])
            dataset.append(
                DatasetEntry(
                    positive=f"{user_tag}Act as described. {asst_tag}{pos_trunc}",
                    negative=f"{user_tag}Act as described. {asst_tag}{neg_trunc}",
                )
            )

    return dataset


# ============================================================================
# Main generation
# ============================================================================

def generate_vectors(model_name: str, output_dir: str, axes: list = None, device: str = None):
    """Generate control vectors for specified personality axes."""

    # Detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    raw_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    raw_model = raw_model.to(device)

    # Get model spec for output directory naming
    arch, n_embd, n_layers = get_model_spec(raw_model, tokenizer)
    print(f"Model: arch={arch}, n_embd={n_embd}, n_layers={n_layers}")

    # Wrap in ControlModel — extract from middle layers (best signal)
    # Use layers from 30% to 70% of total depth
    layer_start = max(1, int(n_layers * 0.3))
    layer_end = min(n_layers - 1, int(n_layers * 0.7))
    layer_ids = list(range(layer_start, layer_end + 1))
    print(f"Extracting from layers {layer_start}-{layer_end} ({len(layer_ids)} layers)")

    model = ControlModel(raw_model, layer_ids)

    # Detect chat template
    user_tag, asst_tag = detect_chat_tags(tokenizer)
    print(f"Chat template: user='{user_tag[:20]}...', asst='{asst_tag[:20]}...'")

    # Output directory: {base}/{arch}_{nEmbd}/
    spec_dir = os.path.join(output_dir, f"{arch}_{n_embd}")
    os.makedirs(spec_dir, exist_ok=True)

    axes_to_generate = axes or list(PERSONALITY_AXES.keys())
    generated = []

    for axis_name in axes_to_generate:
        if axis_name not in PERSONALITY_AXES:
            print(f"Warning: Unknown axis '{axis_name}', skipping")
            continue

        axis_config = PERSONALITY_AXES[axis_name]
        print(f"\n{'='*60}")
        print(f"Generating '{axis_name}' vector...")
        print(f"  {axis_config['description']}")

        dataset = create_dataset(axis_config, tokenizer, user_tag, asst_tag)
        print(f"  Using {len(dataset)} contrastive pairs")

        # Train the control vector
        model.reset()
        vector = ControlVector.train(model, tokenizer, dataset)

        # Export as GGUF (compatible with llama.cpp)
        output_path = os.path.join(spec_dir, f"{axis_name}.gguf")
        vector.export_gguf(output_path)
        file_size = os.path.getsize(output_path)
        print(f"  Saved: {output_path} ({file_size / 1024:.1f} KB)")
        generated.append(axis_name)

    # Write a manifest for reference
    manifest_path = os.path.join(spec_dir, "manifest.json")
    manifest = {
        "model": model_name,
        "architecture": arch,
        "n_embd": n_embd,
        "n_layers": n_layers,
        "layers_used": f"{layer_start}-{layer_end}",
        "axes": {
            name: {
                "file": f"{name}.gguf",
                "description": PERSONALITY_AXES[name]["description"],
            }
            for name in generated
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    # Print deployment instructions
    print(f"\n{'='*60}")
    print("DEPLOYMENT:")
    print(f"  Copy '{spec_dir}/' to device at:")
    print(f"    <filesDir>/control_vectors/{arch}_{n_embd}/")
    print(f"\n  Files to copy:")
    for name in generated:
        print(f"    {name}.gguf")
    print(f"    manifest.json")
    print(f"\n  The PersonaEditorScreen sliders will automatically")
    print(f"  use these vectors when a {arch} model is loaded.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate personality steering vectors for GGUF models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-3B-Instruct",
        help="HuggingFace model name or path (default: SmolLM2-3B-Instruct)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./vectors",
        help="Output directory for vector files (default: ./vectors)",
    )
    parser.add_argument(
        "--axes",
        type=str,
        nargs="*",
        default=None,
        help="Specific axes to generate (default: all). Options: warmth, energy, humor, formality, verbosity, emotion",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect). Options: cuda:0, mps, cpu",
    )
    parser.add_argument(
        "--list-axes",
        action="store_true",
        help="List available personality axes and exit",
    )

    args = parser.parse_args()

    if args.list_axes:
        print("Available personality axes:")
        for name, config in PERSONALITY_AXES.items():
            print(f"  {name}: {config['description']}")
        return

    generate_vectors(args.model, args.output, args.axes, args.device)


if __name__ == "__main__":
    main()
