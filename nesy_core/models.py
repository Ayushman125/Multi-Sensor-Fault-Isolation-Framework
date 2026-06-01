import torch
import torch.nn as nn


class KITTIVisionPerceptionNet(nn.Module):
    def __init__(self):
        super(KITTIVisionPerceptionNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 12)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return torch.softmax(logits, dim=1)


class KITTINeuroSymbolicIsolator(nn.Module):
    def __init__(self, visual_perception_network, device):
        super(KITTINeuroSymbolicIsolator, self).__init__()
        self.perception = visual_perception_network
        self.device = device

    def forward(self, img_cam_t, img_cam_t1, p_odo, p_speed):
        batch_size = img_cam_t.size(0)
        temporal_cam = torch.cat([img_cam_t, img_cam_t1], dim=1)

        p_cam = self.perception(temporal_cam)
        joint_tensor = torch.einsum('bi,bj,bk->bijk', p_cam, p_odo, p_speed)
        isolated_state_output = torch.zeros(batch_size, 10).to(self.device)

        for i in range(10):
            for j in range(10):
                for k in range(10):
                    if i == j == k:
                        isolated_state_output[:, i] += joint_tensor[:, i, j, k]
                    elif j == k and i != j:
                        isolated_state_output[:, j] += joint_tensor[:, i, j, k]
                    elif i == k and i != j:
                        isolated_state_output[:, i] += joint_tensor[:, i, j, k]
                    elif i == j and i != k:
                        isolated_state_output[:, i] += joint_tensor[:, i, j, k]
                    else:
                        isolated_state_output[:, i] += (1.0 / 10.0) * joint_tensor[:, i, j, k]

        return torch.log(isolated_state_output + 1e-8)
