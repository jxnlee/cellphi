import os
import json

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "scale": {
        "px": 370,
        "um": 1800,
    },
    "resolution": {
        "width": 2048,
        "height": 1536
    }
}

def load_config():
    """Loads configuration from JSON or creates a new one with defaults."""
    if not os.path.exists(CONFIG_FILE):
        print("-> Configuration file not found. Creating a default config.json.")
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading config: {e}. Using defaults.")
        return DEFAULT_CONFIG
    
def save_config(config_data):
    """Saves configuration to JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)
    print(f"-> Configuration saved to {CONFIG_FILE}")


def update_config(config):
    """Allows the user to interactively update the configuration."""
    print("\n=== Update Configuration ===")
    
    # Update Pixel Scale
    current_um = config["scale"]["um"]
    new_um = input(f"Enter micrometer distance (Current: {current_um}): ")
    if new_um.strip():
        config["scale"]["um"] = float(new_um)

    current_px = config["scale"]["px"]
    new_px = input(f"Enter pixel distance (Current: {current_px}): ")
    if new_px.strip():
        config["scale"]["px"] = float(new_px)
    
    current_res = config["resolution"]
    new_width = input(f"Enter target width in pixels (Current: {current_res['width']}): ")
    if new_width.strip():
        config["resolution"]["width"] = int(new_width)
    new_height = input(f"Enter target height in pixels (Current: {current_res['height']}): ")
    if new_height.strip():
        config["resolution"]["height"] = int(new_height)
    
    save_config(config)
    return config