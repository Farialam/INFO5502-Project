from torch.utils.data import DataLoader, Dataset
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import pandas as pd
import os
from tkinter import filedialog
import pickle

class TripletNetwork(torch.nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.feature_extractor = models.resnet101(pretrained=True)
        self.feature_extractor.fc = torch.nn.Linear(2048, 128)  # Output: 128-dim feature vector
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return x

# Load the model and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grouped = pd.read_excel(r'best_model_scores.xlsx', sheet_name='Grouped Model Counts')
grouped['category'] = grouped['category'].fillna(method='ffill')
max_scores = grouped.groupby('category')['score'].idxmax()
best_models_for_each_category = grouped.loc[max_scores, 'model']

best_model_dict = {category: model for category, model in zip(grouped['category'].unique(), best_models_for_each_category)}    

class SketchApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Sketch to Artwork")
        self.port = 5000  

        # Frames for Layout
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack()

        self.button_frame = tk.Frame(master)
        self.button_frame.pack()

        # Input and Output Canvas
        self.sketch_canvas = tk.Canvas(self.canvas_frame, bg="white", width=512, height=512)
        self.sketch_canvas.grid(row=0, column=0, padx=10, pady=10)

        self.output_canvas = tk.Canvas(self.canvas_frame, bg="lightgray", width=512, height=512)
        self.output_canvas.grid(row=0, column=1, padx=10, pady=10)

        # Drop-down for Category
        self.category_var = tk.StringVar(master)
        self.category_var.set(list(best_model_dict.keys())[0])  # Default category

        self.category_dropdown = tk.OptionMenu(self.button_frame, self.category_var, *best_model_dict.keys())
        self.category_dropdown.pack(side=tk.LEFT, padx=5)

        # Buttons in a Horizontal Layout
        self.upload_button = tk.Button(self.button_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.button = tk.Button(self.button_frame, text="Find Closest Artwork", command=self.present_output)
        self.button.pack(side=tk.LEFT, padx=5)

        # Sketching Setup
        self.sketch_canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("RGB", (512, 512), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.uploaded_image = None
        self.model = TripletNetwork().to(device)

    def load_model_for_category(self, category):
        model_checkpoint_path = fr"D:\extracted_models\triplet_model_epoch_{best_model_dict[category].split('_')[-1]}_category_{category}.pth"
        print(f"Loading model for category: {category} from {model_checkpoint_path}")
        self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=device), strict=False)
        self.model.eval()

    def paint(self, event):
        x, y = event.x, event.y
        self.sketch_canvas.create_oval(x, y, x + 5, y + 5, fill="black", outline="black")
        self.draw.ellipse((x, y, x + 5, y + 5), fill="black", outline="black")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.uploaded_image = Image.open(file_path).resize((512, 512))
            tk_img = ImageTk.PhotoImage(self.uploaded_image)
            self.sketch_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            self.sketch_canvas.image = tk_img

    def clear_canvas(self):
        self.sketch_canvas.delete("all")
        self.output_canvas.delete("all")
        self.image = Image.new("RGB", (512, 512), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.uploaded_image = None

    def present_output(self):
        category = self.category_var.get()
        embeddings_file = os.path.join('embeddings_folder', f"{category}_embeddings.pt")
        filenames_file = os.path.join('embeddings_folder', f"{category}_filenames.pkl")

        self.load_model_for_category(category)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        artwork_embeddings = torch.load(embeddings_file)
        with open(filenames_file, "rb") as f:
            artwork_filenames = pickle.load(f)

        img_array = self.uploaded_image if self.uploaded_image else self.image

        input_image_tensor = transform(img_array).unsqueeze(0).to(device)
        with torch.no_grad():
            input_embedding = self.model(input_image_tensor)
            distances = torch.norm(artwork_embeddings - input_embedding.cpu(), dim=1)
            closest_idx = torch.argmin(distances).item()

        closest_artwork_path = artwork_filenames[closest_idx]
        print(f"Closest artwork path: {closest_artwork_path}")
        self.display_output_artwork(closest_artwork_path)

    def display_output_artwork(self, artwork_path):
        generated_artwork_img = Image.open(artwork_path).resize((512, 512))
        tk_img = ImageTk.PhotoImage(generated_artwork_img)
        self.output_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        self.output_canvas.image = tk_img

    def visualize_results(self, input_image, generated_artwork_path):
        input_img = input_image
        generated_artwork_img = Image.open(generated_artwork_path)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(input_img)
        axes[0].set_title('Input Sketch')
        axes[0].axis('off')
        axes[1].imshow(generated_artwork_img)
        axes[1].set_title('Generated Artwork')
        axes[1].axis('off')
        plt.show()

# ---- Dataset and Loader Functions ----
class SketchyDataset(Dataset):
    def __init__(self, artwork_path, category, transform=None):
        self.artworks, self.artwork_names = load_artworks_only(artwork_path, category)
        self.transform = transform

    def __len__(self):
        return len(self.artworks)

    def __getitem__(self, idx):
        artwork = self.artworks[idx]
        artwork_name = self.artwork_names[idx]
        if self.transform:
            artwork = self.transform(artwork)
        artwork = np.transpose(artwork, (2, 0, 1))
        return torch.tensor(artwork, dtype=torch.float32), artwork_name

def load_artworks_only(artwork_path, category, size=(256, 256)):
    images, image_paths = [], []
    category_paths = [os.path.join(artwork_path, subdir, category) for subdir in os.listdir(artwork_path)]
    for category_path in category_paths:
        if os.path.isdir(category_path):
            for filename in sorted(os.listdir(category_path)):
                img_path = os.path.join(category_path, filename)
                if img_path.endswith(('.jpg', '.png')):
                    img = Image.open(img_path).convert('RGB').resize(size)
                    img_array = (np.array(img) / 127.5) - 1.0
                    images.append(img_array)
                    image_paths.append(img_path)
    return images, image_paths

def get_dataloader(batch_size, artwork_path, category):
    dataset = SketchyDataset(artwork_path, category)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def compute_embeddings_for_artwork(category, dataloader, model, device):
    artwork_embeddings, artwork_names = [], []
    model.eval()
    with torch.no_grad():
        for positive,positive_name in dataloader:
            positive = positive.to(device)
            embeddings = model(positive)
            artwork_embeddings.append(embeddings.cpu())
            artwork_names.append(list(positive_name))
    return torch.cat(artwork_embeddings, dim=0), [name for sublist in artwork_names for name in sublist]
# ---- Main Application ----
root = tk.Tk()
app = SketchApp(root)
root.mainloop()

