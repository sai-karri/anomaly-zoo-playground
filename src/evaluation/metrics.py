from sklearn.metrics import roc_auc_score
import torch
import mlflow
from pathlib import Path


def evaluation(model, test_loader, device, save_dir="results"):

    model.eval()
    all_scores  = []
    all_labels  = []

    best_score  = -1
    best_amap   = None
    best_image  = None
    best_label  = None

    with torch.no_grad():
        for test_image in test_loader:
            image  = test_image['image'].to(device)
            label  = test_image['label']
            anomaly_score, amap = model.predict(image)

            all_scores.append(anomaly_score.item())
            all_labels.extend(label.numpy())

            # track best defective image
            if label.item() == 1 and anomaly_score.item() > best_score:
                best_score = anomaly_score.item()
                best_amap  = amap.detach().cpu()
                best_image = image.detach().cpu()
                best_label = label

    roc_auc = roc_auc_score(all_labels, all_scores)

    # save best heatmap and log to mlflow
    if best_image is not None:
        from src.visualization.viz import show_heatmap_patchcore
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        heatmap_path = f"{save_dir}/best_heatmap.png"
        show_heatmap_patchcore(
            image=best_image,
            amap=best_amap,
            label=best_label,
            save_path=heatmap_path
        )
        try:
            mlflow.log_artifact(heatmap_path)
            mlflow.log_metric("best_anomaly_score", best_score)
        except Exception:
            pass  # mlflow may not be active in evaluate.py

    return roc_auc