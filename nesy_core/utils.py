import matplotlib.pyplot as plt
import torch


def apply_environmental_lens_blur(images, blur_intensity=1.5):
    """Simulates severe lens mud, salt crusting, or heavy storm occlusion."""
    corrupted_tensors = images + blur_intensity * torch.randn_like(images)
    return torch.clamp(corrupted_tensors, -1.0, 1.0)


def generate_and_save_plot(scenarios, metrics, filename='kitti_realworld_fault_isolation.png'):
    plt.figure(figsize=(10, 5))
    color_palette = ['#0f3410', '#1c3b57', '#7a1f1f']
    safety_bars = plt.bar(scenarios, metrics, color=color_palette, edgecolor='black', width=0.35)

    plt.ylabel('Vehicle State Estimation Accuracy (%)', fontweight='bold', fontsize=10)
    plt.title('Neuro-Symbolic Joint Telemetry Fault Isolation Robustness (KITTI Dataset)', fontsize=11, fontweight='bold')
    plt.ylim(0, 115)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in safety_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 2, f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
