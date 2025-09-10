# setup_folders.py
import os

# --- CONFIGURATION ---
# Add or remove class names here to match your dataset
CLASS_NAMES = [
    "pothole",
    "garbage_dump",
    "waterlogging",
    "broken_streetlight",
    "road_damage",
    "fallen_tree",
    "other_issue" # A general fallback category
]

BASE_DIR = "dataset"
SUB_DIRS = ["train", "val"]

def create_dir_structure():
    """Creates the nested directory structure for the image dataset."""
    print("Setting up directory structure...")
    for sub_dir in SUB_DIRS:
        for class_name in CLASS_NAMES:
            # Construct the full path: e.g., dataset/train/pothole
            path = os.path.join(BASE_DIR, sub_dir, class_name)
            
            # Create the directory, ignoring errors if it already exists
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
    
    print("\nDirectory structure created successfully! ✅")
    print("You can now add your .jpg and .png images to the appropriate folders.")

if __name__ == "__main__":
    create_dir_structure()