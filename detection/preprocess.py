import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'datasets', 'canadian_animals', 'images_and_urls', 'images_and_urls', 'images')
base_dest_dir = os.path.join(base_dir, 'datasets', 'canadian_animals', 'preprocessed')
train_dir = os.path.join(base_dest_dir, 'train')
val_dir = os.path.join(base_dest_dir, 'val')
# Create train and val directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get list of all classes (subdirectories in data_dir)
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Split ratio for training and validation
train_ratio = 0.8

# Process each class
for cls in classes:
    class_dir = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    
    # Split images into training and validation sets
    train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)
    
    # Create class directories in train and val folders
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    
    # Copy training images
    for img in train_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(train_dir, cls, img)
        shutil.copy(src, dst)
    
    # Copy validation images
    for img in val_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(val_dir, cls, img)
        shutil.copy(src, dst)

print("Dataset preprocessing complete. Training and validation sets created.")