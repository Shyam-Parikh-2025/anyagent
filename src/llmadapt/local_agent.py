import os, json, platform, glob


# Find the local path to a model based on its provider and name.
def find_local_model(provider: str,model_name: str) -> str:
    provider = provider.lower()
    home = os.path.expanduser("~")

    # 1. OLLAMA TRACK (Your original logic)
    if provider == "ollama":
        return find_ollama_model_blob(model_name)

    # 2. LM STUDIO TRACK
    elif provider == "lm-studio" or provider == "custom":
        return find_lm_studio_custom_model(home, model_name)

    # 3. HUGGING FACE / VLLM TRACK (The Safetensors Goldmine)
    elif provider == "vllm" or "hf" in provider:
        return find_vllm_hf_model(home, model_name)

    raise FileNotFoundError(f"Could not automatically track down weights for '{model_name}' under provider '{provider}'.")

def find_ollama_model_blob(model_name: str) -> str:
    """
    Automatically tracks down the underlying binary weight file (blob) 
    for an installed Ollama model name.
    """
    # 1. Resolve the cross-platform path to the local Ollama models directory
    home = os.path.expanduser("~")
    if platform.system() == "Windows":
        base_path = os.path.join(home, ".ollama", "models")
    elif platform.system() == "Linux":
        base_path = "/usr/share/ollama/.ollama/models"
        if not os.path.exists(base_path):
            base_path = os.path.join(home, ".ollama", "models")
    else: # macOS
        base_path = os.path.join(home, ".ollama", "models")

    # 2. Standardize the model naming tag matching Ollama's manifest folders
    # e.g., "llama3.1" -> "llama3.1:latest", "mistral:7b" -> "mistral:7b"
    if ":" not in model_name:
        name, tag = model_name, "latest"
    else:
        name, tag = model_name.split(":", 1)

    # 3. Climb through the content-addressable manifest structure
    manifest_file = os.path.join(
        base_path, "manifests", "registry.ollama.ai", "library", name, tag
    )

    if not os.path.exists(manifest_file):
        raise FileNotFoundError(
            f"Model file matching configuration tag '{model_name}' not found locally inside Ollama registry registry paths."
        )

    # 4. Parse the manifest JSON file to read the layer digest hash
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Find the specific layer object mapped to the core weights array
    model_digest = None
    for layer in manifest_data.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            model_digest = layer.get("digest")
            break

    if not model_digest:
        raise ValueError(f"Manifest for '{model_name}' doesn't contain a valid model weights layer field.")

    # 5. Map the hash digest token string back into the actual local blob filename
    # Hashes appear inside manifest as "sha256:abc123xyz..."
    # Windows converts the colon partition into a hyphen delimiter ("sha256-abc123xyz...")
    hash_prefix, hash_body = model_digest.split(":", 1)
    
    if platform.system() == "Windows":
        blob_filename = f"{hash_prefix}-{hash_body}"
    else:
        blob_filename = f"{hash_prefix}-{hash_body}" # Modern updates lean on hyphen formats uniformly

    final_blob_path = os.path.join(base_path, "blobs", blob_filename)
    
    if not os.path.exists(final_blob_path):
        raise FileNotFoundError(f"Underlying model binary file missing at target reference path location: {final_blob_path}")

    return final_blob_path

def find_lm_studio_custom_model(home, model_name: str) -> str:
    """
    Automatically tracks down the underlying GGUF weight file (blob) 
    for an installed LM Studio or custom model name.
    """
    # LM Studio mirrors Hugging Face repo naming structures layout
        # e.g., if model_name is "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
    lm_studio_path = os.path.join(home, ".lmstudio", "models", model_name)
    
    if os.path.exists(lm_studio_path):
        # Find the first .gguf file inside that model's folder
        gguf_files = glob.glob(os.path.join(lm_studio_path, "*.gguf"))
        if gguf_files:
            return gguf_files[0] # Returns the raw GGUF file directly!
    raise FileNotFoundError(f"No GGUF weights found for '{model_name}' in LM Studio directory.")

def find_vllm_hf_model(home, model_name: str) -> str:
    """
    Automatically tracks down the underlying GGUF weight file (blob) 
    for an installed VLLM or HUGGING FACE model name.
    """
    # Hugging Face caches models as snapshots under 'models--user--repo'
    # e.g., meta-llama/Llama-3-8b -> models--meta-llama--Llama-3-8b
    hf_repo_folder = f"models--{model_name.replace('/', '--')}"
    hf_cache_path = os.path.join(home, ".cache", "huggingface", "hub", hf_repo_folder, "snapshots")
    
    if os.path.exists(hf_cache_path):
        # Grab the latest downloaded snapshot folder
        snapshots = sorted(os.listdir(hf_cache_path))
        if snapshots:
            latest_snapshot = os.path.join(hf_cache_path, snapshots[-1])
            # This directory contains the raw .safetensors files your library wants!
            return latest_snapshot
    raise FileNotFoundError(f"No Hugging Face cached snapshot found for '{model_name}'.")

