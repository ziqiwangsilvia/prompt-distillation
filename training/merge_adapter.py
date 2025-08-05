from pathlib import Path

from core import ADAPTER_PATH, MODEL_PATH, BASE_PATH
from core.llm import LLM

def merge_adapter(
    adapter_path: Path,
    save_path: Path
) -> None:
    """
    Merge the adapter at adapter_path into the model and save to save_path.
    """
    llm = LLM.from_adapter(adapter_path)
    llm.load_model()
    llm.model.save_pretrained(save_path)
    llm.tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}.")

def main(
    adapter_name: str,
    merged_checkpoints: str = "merged_checkpoints",
    project_name: str = "huggingface",
) -> None:
    """
    Merge a LoRA adapter (from checkpoints/{project_name}/{adapter_name}) and save to merged_checkpoints/{adapter_name}.
    """
    adapter_path = BASE_PATH / "checkpoints" / project_name / adapter_name
    save_path = Path(merged_checkpoints) / adapter_name

    print(f"Adapter path: {adapter_path}")
    print(f"Save path: {save_path}")
    merge_adapter(adapter_path, save_path)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

