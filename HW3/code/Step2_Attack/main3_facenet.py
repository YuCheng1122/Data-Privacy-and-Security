import argparse
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import InceptionResnetV1
import torch
import torch.nn as nn
import numpy as np

# ---------- CONFIG ----------
CLEAR_DIR = "./celeba_original/"
BLUR_DIR = "./blurred_images_50/"
LABEL_FILE = "./identity_CelebA.txt"
EMBEDDING_SAVE_PATH = "./facenet_model/facenet_embeddings_blurred_50.npz"
CLASSIFIER_SAVE_PATH = "./facenet_model/knn_classifier_blurred_50.pkl"
IMG_SIZE = 160  # FaceNet default input size
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_SAMPLES_PER_ID = 20
# ----------------------------

class FaceDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["identity"]

def load_and_filter_identities(label_file, img_dir, label2id=None):
    df = pd.read_csv(label_file, sep=' ', header=None, names=["filename", "identity"])
    df = df[df["filename"].isin(os.listdir(img_dir))]

    original_total = len(df)
    original_classes = df["identity"].nunique()

    valid_ids = df["identity"].value_counts()
    valid_ids = valid_ids[valid_ids >= MIN_SAMPLES_PER_ID].index
    filtered_df = df[df["identity"].isin(valid_ids)]

    removed_total = original_total - len(filtered_df)
    removed_classes = original_classes - filtered_df["identity"].nunique()

    if label2id is None:
        label2id = {label: idx for idx, label in enumerate(sorted(filtered_df["identity"].unique()))}
    filtered_df = filtered_df.copy()
    filtered_df["identity"] = filtered_df["identity"].map(label2id)

    print(f"Images from: {img_dir}")
    print(f"Original images: {original_total}, classes: {original_classes}")
    print(f"Filtered out:    {removed_total} images, {removed_classes} classes (< {MIN_SAMPLES_PER_ID} images)")
    print(f"Remaining:       {len(filtered_df)} images, {len(label2id)} classes\n")

    return filtered_df, label2id

def prepare_datasets(train_dir, test_dir):
    train_df, label2id = load_and_filter_identities(LABEL_FILE, train_dir)
    test_df, _ = load_and_filter_identities(LABEL_FILE, test_dir, label2id=label2id)

    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["identity"], random_state=42)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    train_set = FaceDataset(train_df, train_dir, transform)
    val_set = FaceDataset(val_df, train_dir, transform)
    test_set = FaceDataset(test_df, test_dir, transform)
    return train_set, val_set, test_set, len(label2id)

def extract_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(DEVICE)
            embs = model(imgs)
            embeddings.append(embs.cpu())
            labels.extend(lbls.numpy())
    return torch.cat(embeddings).numpy(), labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', choices=['clear', 'blurred'], required=True, help="Directory for training images")
    parser.add_argument('--test_dir', choices=['clear', 'blurred'], required=True, help="Directory for testing images")
    args = parser.parse_args()

    train_path = CLEAR_DIR if args.train_dir == 'clear' else BLUR_DIR
    test_path = CLEAR_DIR if args.test_dir == 'clear' else BLUR_DIR

    train_set, val_set, test_set, num_classes = prepare_datasets(train_path, test_path)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load pretrained FaceNet model
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    # Extract embeddings
    print("Extracting embeddings...")
    X_train, y_train = extract_embeddings(facenet, train_loader)
    X_val, y_val = extract_embeddings(facenet, val_loader)
    X_test, y_test = extract_embeddings(facenet, test_loader)

    # Save embeddings
    np.savez(EMBEDDING_SAVE_PATH, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    print(f"Embeddings saved to {EMBEDDING_SAVE_PATH}")

    # Train and save classifier
    print("Training KNN classifier on embeddings...")
    knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
    knn.fit(X_train, y_train)
    joblib.dump(knn, CLASSIFIER_SAVE_PATH)
    print(f"KNN classifier saved to {CLASSIFIER_SAVE_PATH}")

    val_acc = knn.score(X_val, y_val)
    test_acc = knn.score(X_test, y_test)

    print(f"Validation Accuracy (KNN on FaceNet embeddings): {val_acc:.4f}")
    print(f"Test Accuracy (KNN on FaceNet embeddings): {test_acc:.4f}")

if __name__ == "__main__":
    main()