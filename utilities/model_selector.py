"""Interactive model selection utility for Kepler LLM Workbench."""

import os
from pathlib import Path
from typing import List, Optional

from common import ROOT


def find_gguf_models(models_dir: Path = None) -> List[Path]:
    """Find all .gguf model files in the models directory."""
    if models_dir is None:
        models_dir = ROOT / "models"
    
    if not models_dir.exists():
        return []
    
    gguf_files = []
    for file_path in models_dir.rglob("*.gguf"):
        if file_path.is_file():
            gguf_files.append(file_path)
    
    return sorted(gguf_files)


def display_models(models: List[Path]) -> None:
    """Display available models in a formatted list."""
    print("\n" + "="*60)
    print("🤖 Available GGUF Models")
    print("="*60)
    
    if not models:
        print("No GGUF models found in the models directory.")
        print(f"Download models to: {ROOT / 'models'}")
        return
    
    for i, model_path in enumerate(models, 1):
        # Get file size in a human-readable format
        size_bytes = model_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb < 1024:
            size_str = f"{size_mb:.1f} MB"
        else:
            size_gb = size_mb / 1024
            size_str = f"{size_gb:.1f} GB"
        
        # Get relative path from models directory
        rel_path = model_path.relative_to(ROOT / "models")
        
        print(f"{i:2d}. {rel_path}")
        print(f"     Size: {size_str}")


def get_user_selection(models: List[Path]) -> Optional[Path]:
    """Get user's model selection through interactive prompt."""
    if not models:
        return None
    
    while True:
        try:
            print(f"\n🔍 Select a model (1-{len(models)}) or 'q' to quit: ", end="")
            user_input = input().strip()
            
            if user_input.lower() == 'q':
                return None
            
            selection = int(user_input)
            if 1 <= selection <= len(models):
                return models[selection - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(models)}")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return None


def select_model_interactive() -> Optional[Path]:
    """Main interactive model selection function."""
    print("🚀 Kepler LLM Workbench - Model Selection")
    
    models = find_gguf_models()
    display_models(models)
    
    if not models:
        return None
    
    selected_model = get_user_selection(models)
    
    if selected_model:
        print(f"\n✅ Selected: {selected_model.name}")
        return selected_model
    else:
        print("\n👋 No model selected. Goodbye!")
        return None


if __name__ == "__main__":
    selected = select_model_interactive()
    if selected:
        print(f"Model path: {selected}")