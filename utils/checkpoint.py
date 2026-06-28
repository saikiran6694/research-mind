import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Fields that cannot be JSON-serialised (Pydantic models, message objects)
# We store primitives only; Pydantic objects are reconstructed on load.
_SKIP_TYPES = type(None).__mro__   # placeholder — we check per-field below

def _topic_slug(topic: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", topic.lower())
    slug = re.sub(r"[\s]+", "_", slug).strip("_")
    return slug[:60]

def _serialise_state(state: dict) -> dict:
    """
    Convert state to a JSON-safe dict.
    Pydantic models → .model_dump(); others via default=str fallback.
    """
    safe = {}
    for k, v in state.items():
        try:
            if hasattr(v, "model_dump"):
                safe[k] = v.model_dump()
            elif isinstance(v, list):
                items = []
                for item in v:
                    if hasattr(item, "model_dump"):
                        items.append(item.model_dump())
                    elif hasattr(item, "__dict__"):
                        items.append(str(item))
                    else:
                        items.append(item)
                safe[k] = items
            else:
                json.dumps(v)   # test serializability
                safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)
    return safe


def save_checkpoint(topic: str, completed_phase: str, state: dict):
    """
    Call this at the END of every node, passing the *new* state.
    `completed_phase` is the phase that just finished (= state["phase"]).
    """
    slug = _topic_slug(topic=topic)
    path = CHECKPOINT_DIR / f"{slug}.json"

    payload = _serialise_state(state=state)
    payload["_checkpoint_phase"]     = completed_phase
    payload["_checkpoint_saved_at"]  = datetime.now().isoformat()
    payload["_checkpoint_topic"]     = topic

    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[Checkpoint] ✓ Saved after phase '{completed_phase}'")


def load_checkpoint(topic: str) -> Optional[dict]:
    """
    Return the saved state dict if a checkpoint exists, else None.
    Caller is responsible for reconstructing Pydantic objects if needed.
    """
    slug = _topic_slug(topic=topic)
    path = CHECKPOINT_DIR / f"{slug}.json"
    if not path.exists():
        return None
    
    data = json.loads(path.read_text())
    saved_at = data.get("_checkpoint_saved_at", "unknown")
    phase    = data.get("_checkpoint_phase", "unknown")
    print(f"[Checkpoint] ↩ Resuming '{topic}' from phase '{phase}' "
          f"(saved {saved_at})")
    return data

def clear_checkpoint(topic: str):
    """Delete checkpoint once the pipeline completes successfully."""
    slug = _topic_slug(topic=topic)
    path = CHECKPOINT_DIR / f"{slug}.json"
    if path.exists():
        path.unlink()
        print(f"[Checkpoint] ✓ Cleared checkpoint for '{topic}'")


def list_checkpoints() -> list[dict]:
    """Return metadata for all saved checkpoints (useful for CLI inspection)."""
    results = []
    for p in CHECKPOINT_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            results.append({
                "file":   p.name,
                "topic":  data.get("_checkpoint_topic"),
                "phase":  data.get("_checkpoint_phase"),
                "saved":  data.get("_checkpoint_saved_at"),
            })
        except Exception:
            pass
    return results
