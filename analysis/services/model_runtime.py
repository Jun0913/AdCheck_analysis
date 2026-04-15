import os
from pathlib import Path


_MODEL_RUNTIME_READY = False


def prepare_model_runtime() -> None:
    """
    Normalize model-related environment variables so KoBERT/NLI downloads and caches
    stay inside the project workspace and do not fail on unrelated system paths.
    """
    global _MODEL_RUNTIME_READY
    if _MODEL_RUNTIME_READY:
        return

    cache_root = Path(os.getenv("HF_HOME", ".cache/huggingface"))
    hub_cache = Path(os.getenv("HF_HUB_CACHE", cache_root / "hub"))
    cache_root.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)

    sslkeylogfile = os.getenv("SSLKEYLOGFILE")
    if sslkeylogfile:
        try:
            sslkey_path = Path(sslkeylogfile)
            sslkey_path.parent.mkdir(parents=True, exist_ok=True)
            with sslkey_path.open("a", encoding="utf-8"):
                pass
        except OSError:
            os.environ.pop("SSLKEYLOGFILE", None)
            print(f"[model_runtime] SSLKEYLOGFILE disabled: {sslkeylogfile}")

    _MODEL_RUNTIME_READY = True
