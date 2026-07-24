from sklearn.metrics import roc_auc_score
import torch


def evaluation(model, test_loader, device):

    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():

        for test_image in test_loader:
            image = test_image['image'].to(device)
            label = test_image['label']
            anomaly_score, amap = model.predict(image)

            all_scores.append(anomaly_score.item())
            all_labels.extend(label.numpy())

        roc_auc = roc_auc_score(all_labels, all_scores)

    return roc_auc



