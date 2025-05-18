import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, mean_squared_error
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ---------- CONFIG ----------
DP_DIRS = {
    "0.1": "./dp_blurred_images_50_eps_01",
    "1": "./dp_blurred_images_50_eps_1",
    "3": "./dp_blurred_images_50_eps_3",
    "5": "./dp_blurred_images_50_eps_5",
}
ORIGINAL_DIR = "./celeba_original"
LABEL_FILE = "./identity_CelebA.txt"
MODEL_PATH = "./resnet_model/resnet18_blurred_50.pt"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_SAMPLES = 20
BATCH_SIZE = 32
# ----------------------------

class FaceDataset(Dataset):
    def __init__(self, df, img_dir, label_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        orig_path = os.path.join(ORIGINAL_DIR, row["filename"])
        image = Image.open(img_path).convert("RGB")
        orig = Image.open(orig_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            orig = self.transform(orig)

        return image, row["identity"], orig

def load_labels(image_dir):
    df = pd.read_csv(LABEL_FILE, sep=" ", header=None, names=["filename", "identity"])
    df = df[df["filename"].isin(os.listdir(image_dir))]
    counts = df["identity"].value_counts()
    valid_ids = counts[counts >= MIN_SAMPLES].index
    df = df[df["identity"].isin(valid_ids)]
    label2id = {label: idx for idx, label in enumerate(sorted(df["identity"].unique()))}
    df["identity"] = df["identity"].map(label2id)
    return df, label2id

def evaluate_model(model, dataloader):
    model.eval()
    preds, trues = [], []
    ssim_scores = []
    mse_scores = []

    with torch.no_grad():
        for imgs, labels, origs in tqdm(dataloader, desc="Evaluating"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            pred = outputs.argmax(1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels.cpu().numpy())

            for i in range(len(origs)):
                orig_img = origs[i].cpu().permute(1, 2, 0).numpy()  
                dp_img = imgs[i].cpu().permute(1, 2, 0).numpy()   

                orig_img = np.clip(orig_img * 255.0, 0, 255).astype(np.uint8)
                dp_img = np.clip((dp_img + 1) * 127.5, 0, 255).astype(np.uint8)

                ssim_score = ssim(orig_img, dp_img, channel_axis=-1, data_range=255)
                mse_score = mean_squared_error(orig_img.flatten(), dp_img.flatten())
                ssim_scores.append(ssim_score)
                mse_scores.append(mse_score)

    acc = accuracy_score(trues, preds)
    return acc, np.mean(ssim_scores), np.mean(mse_scores)

def main():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    all_results = []

    for eps_tag, dp_dir in DP_DIRS.items():
        print(f"\nTesting images with ε = {eps_tag}...")

        df, label2id = load_labels(dp_dir)
        dataset = FaceDataset(df, dp_dir, label2id, transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(label2id))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)

        acc, avg_ssim, avg_mse = evaluate_model(model, loader)
        print(f"ε={eps_tag} | Accuracy={acc:.4f}, SSIM={avg_ssim:.4f}, MSE={avg_mse:.2f}")

        all_results.append({
            "epsilon": eps_tag,
            "accuracy": acc,
            "ssim": avg_ssim,
            "mse": avg_mse
        })

    # Save results to CSV
    result_df = pd.DataFrame(all_results)
    result_df.to_csv("resnet_dp_summary_pixelscale.csv", index=False)
    print("\nResults saved to resnet_dp_summary_pixelscale.csv")

if __name__ == "__main__":
    main()
