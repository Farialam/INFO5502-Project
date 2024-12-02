
import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pickle

# ---- Step 1: Dataset Class ----
class SketchyDataset(Dataset):
    def __init__(self, artwork_path, category, transform=None):
        self.artworks, self.artwork_names = load_artworks_only(artwork_path, category)
        self.transform = transform

    def __len__(self):
        return len(self.artworks)

    def __getitem__(self, idx):
        artwork = self.artworks[idx]
        artwork_name = self.artwork_names[idx]  # Retrieve the image filename
        if self.transform:
            artwork = self.transform(artwork)
        artwork = np.transpose(artwork, (2, 0, 1))  # Change to [channels, height, width]
        return torch.tensor(artwork, dtype=torch.float32), artwork_name  # Return both artwork and filename


# ---- Step 2: Load Artwork Images Only ----
def load_artworks_only(artwork_path, category, size=(256, 256)):
    images = []
    image_paths = []
    category_paths = [os.path.join(artwork_path, subdir, category) for subdir in os.listdir(artwork_path)]
    
    for category_path in category_paths:
        if os.path.isdir(category_path):
            for filename in sorted(os.listdir(category_path)):
                img_path = os.path.join(category_path, filename)
                if img_path.endswith('.jpg') or img_path.endswith('.png'):
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(size)
                    img_array = (np.array(img) / 127.5) - 1.0  # Normalize to [-1, 1]
                    images.append(img_array)
                    image_paths.append(img_path)
                    
    return images, image_paths


# ---- Step 3: DataLoader Preparation ----
def get_dataloader(batch_size, artwork_path, category):
    """
    Prepares a DataLoader for the given category.
    """
    transform = None  # Add any transformations here
    dataset = SketchyDataset(artwork_path=artwork_path, category=category, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---- Step 4: Compute Embeddings ----
def compute_embeddings_for_artwork(category, dataloader, device):
    """
    Compute embeddings for artworks in a specific category.
    Placeholder function: Replace with your actual embedding computation logic.
    """
    embeddings = []
    filenames = []

    for batch in dataloader:
        images, names = batch
        images = images.to(device)

        # Placeholder for actual model
        with torch.no_grad():
            # Example: Random embeddings, replace this with actual model inference
            batch_embeddings = torch.randn(images.size(0), 128)  # Example embedding size 128

        embeddings.append(batch_embeddings)
        filenames.extend(names)

    embeddings = torch.cat(embeddings, dim=0)  # Combine all embeddings
    return embeddings, filenames


# ---- Step 5: Save Embeddings ----
def save_embeddings(embeddings, filenames, category, save_path):
    """
    Save embeddings and filenames to a specified folder.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save embeddings as a .pt file
    embeddings_file = os.path.join(save_path, f"{category}_embeddings.pt")
    torch.save(embeddings, embeddings_file)

    filenames_file = os.path.join(save_path, f"{category}_filenames.pkl")
    with open(filenames_file, "wb") as f:
            pickle.dump(artwork_filenames, f)
    
    # # Save filenames as a JSON file
    # filenames_file = os.path.join(save_path, f"{category}_filenames.json")
    # with open(filenames_file, 'w') as f:
    #     json.dump(filenames, f)
    
    print(f"Saved embeddings and filenames for category '{category}' to {save_path}.")


# ---- Step 6: Main Script ----
if __name__ == "__main__":
    # Folder and categories setup
    folder_location = r'C:\Users\dhana\Desktop\UNT Course\Sem1\5502\Project work\rendered_256x256\256x256\photo'  # Path to your dataset
    categories = list(set(['_'.join(model.split('.')[0].split('_')[5:]) for model in os.listdir(r'D:\extracted_models')]))
    # categories = ["cat1", "cat2", "cat3"]  # Replace with your actual category names
    save_path = r"embeddings_folder"  # Directory to save embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    # Loop through each category, compute embeddings, and save them
    for category in categories:
        print(f"Processing category: {category}")
        
        # Prepare the dataloader for the current category
        dataloader = get_dataloader(
            batch_size=16,
            artwork_path=folder_location,
            category=category
        )
        
        # Compute embeddings and filenames for the current category
        artwork_embeddings, artwork_filenames = compute_embeddings_for_artwork(category, dataloader, device)
        
        # Save the embeddings and filenames
        save_embeddings(artwork_embeddings, artwork_filenames, category, save_path)

    print("All embeddings have been computed and saved.")
