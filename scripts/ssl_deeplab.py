import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.deeplab.model import get_deeplab
from models.deeplab.dataset import SemanticSegmentationDataset

# ------------------------
# SSL configuration
# ------------------------
IGNORE_INDEX = 255
CONF_THR = 0.9
EPOCHS = 50
BATCH = 4
LR = 1e-4


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ------------------------
    # Datasets
    # ------------------------
    train_ds = SemanticSegmentationDataset(
        base_dir="dataset",
        split="train",
        unlabeled=False
    )

    ssl_ds = SemanticSegmentationDataset(
        base_dir="dataset",
        split="ssl",
        unlabeled=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    ssl_loader = DataLoader(ssl_ds, batch_size=BATCH, shuffle=True)

    # ------------------------
    # Model
    # ------------------------
    model = get_deeplab(num_classes=8).to(device)
    model.load_state_dict(
        torch.load(
            "scripts/model_weights/deeplab_initial.pth",
            map_location=device
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # ------------------------
    # Training loop
    # ------------------------
    model.train()
    ssl_iter = iter(ssl_loader)

    for epoch in range(EPOCHS):
        for x_l, y_l in train_loader:
            x_l, y_l = x_l.to(device), y_l.to(device)

            # Supervised loss
            logits_l = model(x_l)["out"]
            loss_sup = criterion(logits_l, y_l)

            # Unlabeled batch
            try:
                x_u = next(ssl_iter)
            except StopIteration:
                ssl_iter = iter(ssl_loader)
                x_u = next(ssl_iter)

            x_u = x_u.to(device)

            # Pseudo-labels
            with torch.no_grad():
                logits_u = model(x_u)["out"]
                probs_u = F.softmax(logits_u, dim=1)
                conf, pseudo = probs_u.max(dim=1)
                pseudo[conf < CONF_THR] = IGNORE_INDEX

            # SSL loss
            logits_u2 = model(x_u)["out"]
            loss_ssl = criterion(logits_u2, pseudo)

            loss = loss_sup + loss_ssl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss={loss.item():.4f} "
            f"(Sup={loss_sup.item():.4f}, SSL={loss_ssl.item():.4f})"
        )

    torch.save(
        model.state_dict(),
        "scripts/model_weights/deeplab_ssl.pth"
    )
    print("✅ Training complete. Model saved as deeplab_ssl.pth")


if __name__ == "__main__":
    main()
