import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from nesy_core.dataset import KITTITemporalVelocityDataset
from nesy_core.models import KITTIVisionPerceptionNet, KITTINeuroSymbolicIsolator
from nesy_core.utils import apply_environmental_lens_blur, generate_and_save_plot


def main():
    parser = argparse.ArgumentParser(description='Neuro-Symbolic AV Telemetry Fault Isolation Framework')
    parser.add_argument('--epochs', type=int, default=3, help='Number of operational training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate parameter')
    parser.add_argument('--batch_size', type=int, default=32, help='Data loader batch step size')
    parser.add_argument('--data_path', type=str, default='./data/kitti', help='Root location for local KITTI folders')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] Starting execution engine on device: {device}')

    full_temporal_set = KITTITemporalVelocityDataset(root_dir=args.data_path, split='training', labeled=True)
    train_size = int(0.8 * len(full_temporal_set))
    val_size = len(full_temporal_set) - train_size
    kitti_train_set, kitti_val_set = random_split(
        full_temporal_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    kitti_test_images = KITTITemporalVelocityDataset(root_dir=args.data_path, split='testing', labeled=False)

    train_loader = DataLoader(kitti_train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(kitti_val_set, batch_size=args.batch_size, shuffle=False)
    test_image_loader = DataLoader(kitti_test_images, batch_size=8, shuffle=False)

    vision_model = KITTIVisionPerceptionNet().to(device)
    kitti_nesy_system = KITTINeuroSymbolicIsolator(vision_model, device).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(kitti_nesy_system.parameters(), lr=args.lr)

    print('\n=== Training Temporal Neuro-Symbolic Safety Architecture ===')
    for epoch in range(args.epochs):
        kitti_nesy_system.train()
        running_loss, correct, total = 0.0, 0, 0
        for temporal_image, p_odo, p_speed, targets in train_loader:
            temporal_image, p_odo, p_speed, targets = (
                temporal_image.to(device),
                p_odo.to(device),
                p_speed.to(device),
                targets.to(device),
            )
            valid_mask = targets >= 0
            if valid_mask.sum().item() == 0:
                continue

            temporal_image = temporal_image[valid_mask]
            p_odo = p_odo[valid_mask]
            p_speed = p_speed[valid_mask]
            targets = targets[valid_mask]

            optimizer.zero_grad()
            img_cam_t = temporal_image[:, :3, :, :]
            img_cam_t1 = temporal_image[:, 3:, :, :]
            log_outputs = kitti_nesy_system(img_cam_t, img_cam_t1, p_odo, p_speed)
            loss = criterion(log_outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(log_outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        print(f'KITTI Epoch {epoch + 1}/{args.epochs} | Navigation Loss: {running_loss / len(train_loader):.4f} | System Integrity: {(correct / total) * 100:.2f}%')

    for param in vision_model.parameters():
        param.requires_grad = False

    print('\n=== Simulating High-Stress On-Road Sensor Casualties ===')
    kitti_nesy_system.eval()
    test_scenarios = ['Nominal Clear Highway', 'Blinded Dashboard Camera (Mud Occlusion)', 'Blinded Dashboard Camera + Broken Odometer']
    test_metrics = []

    for scenario in test_scenarios:
        correct, total = 0, 0
        with torch.no_grad():
            for temporal_image, p_odo, p_speed, targets in val_loader:
                temporal_image, p_odo, p_speed, targets = (
                    temporal_image.to(device),
                    p_odo.to(device),
                    p_speed.to(device),
                    targets.to(device),
                )
                valid_mask = targets >= 0
                if valid_mask.sum().item() == 0:
                    continue

                temporal_image = temporal_image[valid_mask]
                p_odo = p_odo[valid_mask]
                p_speed = p_speed[valid_mask]
                targets = targets[valid_mask]

                img_cam_t = temporal_image[:, :3, :, :]
                img_cam_t1 = temporal_image[:, 3:, :, :]
                if scenario == 'Blinded Dashboard Camera (Mud Occlusion)':
                    img_cam_t = apply_environmental_lens_blur(img_cam_t)
                    img_cam_t1 = apply_environmental_lens_blur(img_cam_t1)
                elif scenario == 'Blinded Dashboard Camera + Broken Odometer':
                    img_cam_t = apply_environmental_lens_blur(img_cam_t)
                    img_cam_t1 = apply_environmental_lens_blur(img_cam_t1)
                    p_odo = torch.softmax(torch.log(p_odo + 1e-8) + 0.75 * torch.randn_like(p_odo), dim=1)

                log_outputs = kitti_nesy_system(img_cam_t, img_cam_t1, p_odo, p_speed)
                preds = torch.argmax(log_outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        safety_accuracy = (correct / total) * 100
        test_metrics.append(safety_accuracy)
        print(f'Scenario: [{scenario:45s}] -> Navigational Safe Tracking: {safety_accuracy:.2f}%')

    print('\n=== Sample Predictions on Unlabeled KITTI Testing Images ===')
    with torch.no_grad():
        for batch_index, (temporal_image, p_odo, p_speed, _) in enumerate(test_image_loader):
            temporal_image, p_odo, p_speed = temporal_image.to(device), p_odo.to(device), p_speed.to(device)
            img_cam_t = temporal_image[:, :3, :, :]
            img_cam_t1 = temporal_image[:, 3:, :, :]
            log_outputs = kitti_nesy_system(img_cam_t, img_cam_t1, p_odo, p_speed)
            preds = torch.argmax(log_outputs, dim=1).cpu().tolist()
            print(f'Testing batch {batch_index + 1}: predicted buckets = {preds[:10]}')
            break

    generate_and_save_plot(test_scenarios, test_metrics)
    print("\n[SUCCESS] Generated final metric asset chart to 'kitti_realworld_fault_isolation.png'")


if __name__ == '__main__':
    main()
