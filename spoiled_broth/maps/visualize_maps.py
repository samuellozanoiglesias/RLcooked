"""
Visualize and store maps from text specifications and sprites

Usage:
    nohup python visualize_maps.py > visualize_maps.log 2>&1 &
"""

from pathlib import Path
from PIL import Image
import glob
import os


def show_map_from_txt(txt_path, output_name):
    """
    Create map visualization from text file specification
    
    Args:
        txt_path: Path to .txt map file
        output_name: Name for output PNG file
    """
    # Read the text map
    with open(txt_path, 'r') as f:
        map_lines = [line.rstrip('\n') for line in f.readlines()]
    
    if not map_lines:
        print(f"Warning: Empty map file {txt_path}")
        return
    
    height = len(map_lines)
    width = len(map_lines[0]) if map_lines else 0
    
    # Create empty canvas
    canvas = Image.new("RGBA", (width * 16, height * 16))
    
    # Character to tile type mapping
    char_to_tile = {
        'W': 'Wall',
        'M': 'Counter',
        'B': 'CuttingBoard',
        'T': 'Dispenser_tomato',
        'P': 'Dispenser_pumpkin',
        'X': 'Dispenser_plate',
        'C': 'Dispenser_cabbage',
        'D': 'Delivery',
        ' ': 'Floor',
        '1': 'Floor',  # Agent 1 starting position (render as floor)
        '2': 'Floor',  # Agent 2 starting position (render as floor)
    }
    
    # Render each tile
    for y, line in enumerate(map_lines):
        for x, char in enumerate(line):
            tile_type = char_to_tile.get(char, 'Floor')
            
            if tile_type not in asset_map:
                print(f"Warning: Unknown tile type '{tile_type}' for char '{char}', using Floor")
                tile_type = 'Floor'
            
            sprite_paths = asset_map[tile_type]['paths']
            
            for i, sprite_path in enumerate(sprite_paths):
                if not sprite_path.exists():
                    print(f"Warning: Sprite not found: {sprite_path}")
                    continue
                
                # Load sprite with alpha channel
                sprite_image = Image.open(sprite_path).convert("RGBA")
                
                # Crop specific regions for dispensers
                if tile_type == "Dispenser_tomato" and i == 1:
                    sprite_image = sprite_image.crop((0, 0, 16, 16))
                elif tile_type == "Dispenser_plate" and i == 1:
                    sprite_image = sprite_image.crop((0, 48, 16, 64))
                elif tile_type == "Dispenser_pumpkin" and i == 1:
                    sprite_image = sprite_image.crop((16, 0, 32, 16))
                elif tile_type == "Dispenser_cabbage" and i == 1:
                    sprite_image = sprite_image.crop((32, 0, 48, 16))
                else:
                    sprite_image = sprite_image.crop((0, 0, 16, 16))
                
                # Paste sprite onto canvas
                canvas.paste(sprite_image, (x * 16, y * 16), sprite_image)
    
    # Save the visualization
    canvas.save(output_name)
    print(f"✓ Created: {output_name}")


asset_map = {
    "Floor": {
        "paths": [Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-floor.png"],
    },
    "Wall": {
        "paths": [Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-wall.png"],
    },
    "Counter": {
        "paths": [Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png"],
    },
    "CuttingBoard": {
        "paths": [
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "cutting-board.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "knife.png",
        ],
    },
    "Dispenser_tomato": {
        "paths": [
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "item-dispenser.png",
        ],
    },
    "Dispenser_plate": {
        "paths": [
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "item-dispenser.png",
        ],
    },
    "Dispenser_pumpkin": {
        "paths": [
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "item-dispenser.png",
        ],
    },
    "Dispenser_cabbage": {
        "paths": [
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "item-dispenser.png",
        ],
    },
    "Delivery": {
        "paths": [Path(__file__).parent.parent / "static" / "sprites" / "world" / "delivery.png"],
    },
}


if __name__ == "__main__":
    # Setup paths
    maps_dir = Path(__file__).parent / "maps_txt"
    output_dir = Path(__file__).parent / "maps_png"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Find all .txt map files (exclude info files)
    txt_files = glob.glob(str(maps_dir / "*.txt"))
    txt_files = [f for f in txt_files if not f.endswith('_info.txt')]
    
    print(f"Found {len(txt_files)} map files")
    print(f"Generating visualizations in: {output_dir}")
    print("=" * 60)
    
    # Process each map file
    for txt_path in sorted(txt_files):
        map_name = Path(txt_path).stem  # Get filename without extension
        output_path = output_dir / f"{map_name}.png"
        
        try:
            show_map_from_txt(txt_path, str(output_path))
        except Exception as e:
            print(f"✗ Error processing {map_name}: {e}")
    
    print("=" * 60)
    print(f"✓ Done! Generated {len(txt_files)} map visualizations")