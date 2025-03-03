# helper functions
import monai
import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_batch(train_loader, val_loader, test_loader):
    batch = monai.utils.first(train_loader)
    images, labels = batch["img"], batch["seg"]
    for i in range(4):
        plt.figure("images")
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(images[i][0], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(labels[i][0], cmap="gray")
        plt.show()

    batch = monai.utils.first(val_loader)
    images, labels = batch["img"], batch["seg"]
    for i in range(4):
        plt.figure("images")
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(images[i][0], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(labels[i][0], cmap="gray")
        plt.show()

    batch = monai.utils.first(test_loader)
    images, labels = batch["img"], batch["seg"]
    for i in range(4):
        plt.figure("images")
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(images[i][0], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(labels[i][0], cmap="gray")
        plt.show()


# Helper func to visualize test results
def visualize_segmentation_results(
    original_image, ground_truth_mask, predicted_mask, alpha=0.5
):
    
    ground_truth_mask = apply_color_map(ground_truth_mask)
    predicted_mask = apply_color_map(predicted_mask)
    
    # Create the figure and the axes
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Display the ground truth mask
    axes[0].imshow(ground_truth_mask, cmap="gray")
    axes[0].set_title("Ground Truth Mask")
    axes[0].axis("off")

    # Display the predicted mask
    axes[1].imshow(predicted_mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    # Display the original image with ground truth mask overlay
    axes[2].imshow(original_image)
    axes[2].imshow(ground_truth_mask, cmap="jet", alpha=alpha)
    axes[2].set_title("Ground Truth Overlay")
    axes[2].axis("off")

    # Display the original image with predicted mask overlay
    axes[3].imshow(original_image)
    axes[3].imshow(predicted_mask, cmap="jet", alpha=alpha)
    axes[3].set_title("Predicted Overlay")
    axes[3].axis("off")

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
def save_val_results(val_gt, val_outputs, save_path):
    batch_size = val_gt.shape[0]
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size*5, 10))

    for i in range(batch_size):
        axes[0, i].imshow(val_gt[i][0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(val_outputs[i][0], cmap="gray")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)


def apply_color_map(seg_mask):
    # Create an empty image with 3 channels (RGB)
    color_image = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)

    # Define colors for each segment value (0: black, 1: red, 2: green, 3: blue)
    colors = {
        0: [0, 0, 0],  # black for background
        1: [255, 0, 0],  # red
        2: [0, 255, 0],  # green
        3: [0, 0, 255],  # blue
        # Add more colors if there are more segments
    }

    # Apply the colors to the image
    for val, color in colors.items():
        color_image[seg_mask == val] = color

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image


def dice_coefficient_per_class(predicted_mask, true_mask, num_classes):
    dice_scores = []
    for cls in range(num_classes):
        # Calculate the intersection and union for the current class
        intersection = ((predicted_mask == cls) & (true_mask == cls)).sum().item()
        predicted_sum = (predicted_mask == cls).sum().item()
        true_sum = (true_mask == cls).sum().item()

        if predicted_sum + true_sum == 0:
            dice = float(
                "nan"
            )  # Handle case where no instances of the class exist in either prediction or ground truth
        else:
            dice = (2 * intersection) / (predicted_sum + true_sum)

        dice_scores.append(dice)

    return dice_scores


def iou_per_class(predicted_mask, true_mask, num_classes):
    iou_scores = []
    for cls in range(num_classes):
        # Calculate the intersection and union for the current class
        intersection = ((predicted_mask == cls) & (true_mask == cls)).sum().item()
        union = ((predicted_mask == cls) | (true_mask == cls)).sum().item()

        if union == 0:
            iou = float(
                "nan"
            )  # Handle case where no instances of the class exist in either prediction or ground truth
        else:
            iou = intersection / union

        iou_scores.append(iou)

    return iou_scores


def precision_per_class(predicted_mask, true_mask, num_classes):
    precision_scores = []
    for cls in range(num_classes):
        tp = ((predicted_mask == cls) & (true_mask == cls)).sum().item()
        fp = ((predicted_mask == cls) & (true_mask != cls)).sum().item()

        if tp + fp == 0:
            precision = float("nan")
        else:
            precision = tp / (tp + fp)

        precision_scores.append(precision)

    return precision_scores


def recall_per_class(predicted_mask, true_mask, num_classes):
    recall_scores = []
    for cls in range(num_classes):
        tp = ((predicted_mask == cls) & (true_mask == cls)).sum().item()
        fn = ((predicted_mask != cls) & (true_mask == cls)).sum().item()

        if tp + fn == 0:
            recall = float("nan")
        else:
            recall = tp / (tp + fn)

        recall_scores.append(recall)

    return recall_scores


def f1_score_per_class(predicted_mask, true_mask, num_classes):
    f1_scores = []
    for cls in range(num_classes):
        precision = precision_per_class(predicted_mask, true_mask, num_classes)[cls]
        recall = recall_per_class(predicted_mask, true_mask, num_classes)[cls]

        if precision + recall == 0:
            f1 = float("nan")
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        f1_scores.append(f1)

    return f1_scores


def accuracy_per_class(predicted_mask, true_mask, num_classes):
    accuracy_scores = []
    for cls in range(num_classes):
        tp = ((predicted_mask == cls) & (true_mask == cls)).sum().item()
        tn = ((predicted_mask != cls) & (true_mask != cls)).sum().item()
        fp = ((predicted_mask == cls) & (true_mask != cls)).sum().item()
        fn = ((predicted_mask != cls) & (true_mask == cls)).sum().item()

        if tp + tn + fp + fn == 0:
            accuracy = float("nan")
        else:
            accuracy = (tp + tn) / (tp + tn + fp + fn)

        accuracy_scores.append(accuracy)
    return accuracy_scores