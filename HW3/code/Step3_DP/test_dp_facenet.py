import os
import torch
import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import pandas as pd

# ---------- CONFIG ----------
DP_DIR = "./dp_blurred_images_50_eps_5"

LABEL_FILE = "./identity_CelebA.txt"
KNN_MODEL_PATH = "./facenet_model/knn_classifier_blurred_50.pkl"
IMG_SIZE = 160
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------

# Load label mapping
def load_filtered_labels(label_file, image_dir, min_samples=20):
    df = pd.read_csv(label_file, sep=" ", header=None, names=["filename", "identity"])
    df = df[df["filename"].isin(os.listdir(image_dir))]
    valid_ids = df["identity"].value_counts()
    valid_ids = valid_ids[valid_ids >= min_samples].index
    df = df[df["identity"].isin(valid_ids)]
    label2id = {label: idx for idx, label in enumerate(sorted(df["identity"].unique()))}
    df["identity"] = df["identity"].map(label2id)
    return df.set_index("filename")["identity"].to_dict(), len(label2id)

# Load images and compute embeddings
def load_images_and_embeddings(image_dir, label_dict, model):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    X, y, used_files = [], [], []
    for fname in os.listdir(image_dir):
        if fname not in label_dict:
            continue
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model(img_tensor).cpu().numpy()[0]
        X.append(embedding)
        y.append(label_dict[fname])
        used_files.append(fname)
    return np.array(X), np.array(y), used_files

def main():
    print("Loading FaceNet model and KNN classifier...")
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    knn = joblib.load(KNN_MODEL_PATH)

    print("Loading labels and images...")
    label_dict, num_classes = load_filtered_labels(LABEL_FILE, DP_DIR)
    X_test, y_test, filenames = load_images_and_embeddings(DP_DIR, label_dict, facenet)

    print("Running recognition test...")
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nRecognition accuracy on DP image set: {acc:.4f}")
    print(f"Total test samples: {len(y_test)}")

    # Save result summary to .txt
    epsilon_tag = DP_DIR.split("_")[-1]
    log_file = f"dp_test_summary_eps_{epsilon_tag}.txt"
    with open(log_file, "w") as f:
        f.write(f"Epsilon tag: {epsilon_tag}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Samples: {len(y_test)}\n")

    print(f"Accuracy summary saved to: {log_file}")

    # Save filenames of evaluated images
    with open("test_filenames.txt", "w") as f:
        for name in filenames:
            f.write(f"{name}\n")

if __name__ == "__main__":
    main()
