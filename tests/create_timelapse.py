"""
create_timelapse.py - Vytvoří timelapse video z vygenerovaných vizualizací.

Použití:
    python create_timelapse.py                    # Vytvoří všechny timelapse
    python create_timelapse.py --type scatter     # Jen scatter plot timelapse
    python create_timelapse.py --fps 10           # Rychlejší video (10 fps)

Výstup:
    stress_test_results/timelapse/
    ├── scatter_timelapse.mp4
    ├── landscape_timelapse.mp4
    └── histogram_timelapse.mp4

Požadavky:
    pip install imageio[ffmpeg]
    # nebo
    pip install opencv-python
"""

import argparse
from pathlib import Path
from typing import List
import re

def get_sorted_images(timelapse_dir: Path, prefix: str) -> List[Path]:
    """Získá seřazené obrázky podle step čísla."""
    pattern = re.compile(rf"{prefix}_step_(\d+)\.png")
    images = []
    
    for img_path in timelapse_dir.glob(f"{prefix}_step_*.png"):
        match = pattern.match(img_path.name)
        if match:
            step_num = int(match.group(1))
            images.append((step_num, img_path))
    
    # Seřaď podle step čísla
    images.sort(key=lambda x: x[0])
    return [img[1] for img in images]


def create_timelapse_imageio(images: List[Path], output_path: Path, fps: int = 5):
    """Vytvoří timelapse pomocí imageio (preferovaná metoda)."""
    try:
        import imageio.v2 as imageio
        
        print(f"  Načítám {len(images)} obrázků...")
        frames = [imageio.imread(str(img)) for img in images]
        
        print(f"  Zapisuji video: {output_path}")
        imageio.mimsave(str(output_path), frames, fps=fps)
        
        return True
    except ImportError:
        return False


def create_timelapse_opencv(images: List[Path], output_path: Path, fps: int = 5):
    """Vytvoří timelapse pomocí OpenCV (alternativa)."""
    try:
        import cv2
        
        print(f"  Načítám {len(images)} obrázků...")
        
        # Načti první obrázek pro rozměry
        first_frame = cv2.imread(str(images[0]))
        height, width, _ = first_frame.shape
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for img_path in images:
            frame = cv2.imread(str(img_path))
            out.write(frame)
        
        out.release()
        print(f"  Uloženo: {output_path}")
        
        return True
    except ImportError:
        return False


def create_gif(images: List[Path], output_path: Path, duration: float = 0.2):
    """Vytvoří GIF jako alternativu k videu."""
    try:
        from PIL import Image
        
        print(f"  Načítám {len(images)} obrázků...")
        frames = [Image.open(img) for img in images]
        
        # Uloží jako GIF
        gif_path = output_path.with_suffix('.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(duration * 1000),
            loop=0
        )
        print(f"  Uloženo: {gif_path}")
        
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Vytvoří timelapse z vizualizací')
    parser.add_argument('--input', type=str, default='stress_test_results/timelapse',
                       help='Adresář s obrázky')
    parser.add_argument('--type', type=str, choices=['scatter', 'landscape', 'histogram', 'all'],
                       default='all', help='Typ vizualizace')
    parser.add_argument('--fps', type=int, default=5, help='Snímků za sekundu')
    parser.add_argument('--format', type=str, choices=['mp4', 'gif'], default='mp4',
                       help='Výstupní formát')
    
    args = parser.parse_args()
    
    timelapse_dir = Path(args.input)
    
    if not timelapse_dir.exists():
        print(f"❌ Adresář {timelapse_dir} neexistuje!")
        print("   Nejprve spusťte stress_test_memory.py pro vygenerování vizualizací.")
        return
    
    print("=" * 60)
    print("🎬 TIMELAPSE CREATOR")
    print("=" * 60)
    
    types_to_process = ['scatter', 'landscape', 'histogram'] if args.type == 'all' else [args.type]
    
    for viz_type in types_to_process:
        images = get_sorted_images(timelapse_dir, viz_type)
        
        if not images:
            print(f"⚠ Žádné obrázky pro {viz_type}")
            continue
        
        print(f"\n📹 Zpracovávám: {viz_type} ({len(images)} snímků)")
        
        output_path = timelapse_dir / f"{viz_type}_timelapse.{args.format}"
        
        if args.format == 'mp4':
            success = create_timelapse_imageio(images, output_path, args.fps)
            if not success:
                success = create_timelapse_opencv(images, output_path, args.fps)
            if not success:
                print("  ⚠ Nelze vytvořit MP4, zkouším GIF...")
                create_gif(images, output_path, 1.0 / args.fps)
        else:
            create_gif(images, output_path, 1.0 / args.fps)
    
    print("\n" + "=" * 60)
    print("✅ Timelapse hotov!")
    print(f"   Výstupy v: {timelapse_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
