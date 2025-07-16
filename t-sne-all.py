import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from networks.model_Copy1 import MainNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd


def extract_features(model, dataloader, device):
    model.eval()
    global_features = []
    local_features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                out_global, out_local = outputs
                global_features.append(out_global.cpu())
                local_features.append(out_local.cpu())
            else:
                global_features.append(outputs.cpu())
                local_features.append(torch.zeros_like(outputs.cpu()))

            labels.append(targets.cpu())

    global_features = torch.cat(global_features)
    local_features = torch.cat(local_features)
    labels = torch.cat(labels)

    return global_features, local_features, labels


def fuse_features(global_feats, local_feats):
    return torch.cat((global_feats, local_feats), dim=1)


def save_all_class_tsne(features, labels, class_names, output_file):
    """降维 + 保存所有类别 t-SNE 坐标到 CSV"""
    print(f"正在进行 PCA + t-SNE 降维，共 {len(features)} 条样本...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate=200)
    embeddings = tsne.fit_transform(features_pca)

    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

    tsne_data = pd.DataFrame({
        'X': embeddings[:, 0],
        'Y': embeddings[:, 1],
        'Label': [class_names[lbl] for lbl in labels_np]
    })
    tsne_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已保存 t-SNE 坐标到: {output_file}")


def load_model(model_path, num_classes, device):
    try:
        model = MainNet(num_classes=num_classes).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        pretrained_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model_dict = model.state_dict()
        matched_params = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched_params)
        model.load_state_dict(model_dict, strict=False)
        print(f"成功加载参数: {len(matched_params)}/{len(model_dict)}")
        return model
    except Exception as e:
        print(f"加载失败: {str(e)}")
        return None


def create_dataloader(data_root, transform):
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    class_names = dataset.classes
    return dataloader, class_names


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_loader, class_names = create_dataloader(r'F:\paper\SRFL-OL\datasets\2884MARC/test', test_transform)

    model_paths = [
        (r"F:\paper\SRFL-OL\512OL\epoch3.pth", False),
        (r"F:\paper\SRFL-OL\best.pth", True)
    ]

    for model_path, is_fusion in model_paths:
        model = load_model(model_path, 2884, device)
        if model is None:
            continue

        global_feats, local_feats, labels = extract_features(model, test_loader, device)

        if is_fusion and local_feats is not None:
            full_features = fuse_features(global_feats, local_feats)
            name_prefix = "our_proposed"
        else:
            full_features = global_feats
            name_prefix = "backbone"

        save_all_class_tsne(
            features=full_features.numpy(),
            labels=labels,
            class_names=class_names,
            output_file=f"{name_prefix}_tsne_all_data.csv"
        )
