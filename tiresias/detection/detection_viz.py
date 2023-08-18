import matplotlib.pyplot as plt
import mmcv
import mmdet.models.detectors
from mmdet.registry import VISUALIZERS
from typing import List, Dict, Any
from tiresias.detection.detection_infer import infer_detection

import warnings
warnings.filterwarnings("ignore", message="__floordiv__ is deprecated")


def generate_pred_plot(img_path: str, inferencer: mmdet.models.detectors.BaseDetector, timing: bool = False, pred_score_thr: float = 0.5) -> Dict[str, Any]:
    """Generate prediction visualization for an image according to a detector.

    Args:
        img_path (str): Path to the input image.
        inferencer (mmdet.models.detectors.BaseDetector): Inference object with detection capabilities.
        timing (bool, optional): Whether to measure inference timing. Defaults to False.

    Returns:
        Dict[str, Any]: Dictionary containing model name, file name, and result plot.
    """
    # Get prediction
    result = infer_detection(img_path=img_path, inferencer=inferencer, timing=timing)
    # Get raw image
    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    # Init visualizer
    inferencer_visualizer = VISUALIZERS.build(inferencer.cfg.visualizer)
    inferencer_visualizer.dataset_meta = inferencer.dataset_meta
    # Show the results
    inferencer_visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
        pred_score_thr=pred_score_thr,  # Default threshold in Openmmlab 0.3
        show=True
    )
    try:
        if inferencer.custom:
            model_name = 'Custom_' + inferencer._get_name()
    except AttributeError:
        model_name = inferencer._get_name()
    return {
        "model_name": model_name,
        "file_name": img_path,
        "result_plot": inferencer_visualizer.get_image()
    }


def display_pred_mosaic(image_dicts_list: List[Dict[str, Any]], width: int = 16, height: int = 6) -> None:
    """Display a mosaic of 2 columns a list of prediction results.

    Args:
        image_dicts_list (List[Dict[str, Any]]): List of dictionaries containing prediction results.
        width (int, optional): Width of the displayed figure. Defaults to 16.
        height (int, optional): Height of the displayed figure. Defaults to 6.

    Raises:
        ValueError: If the list contains only one element.
    """
    if len(image_dicts_list) < 2:
        raise ValueError("The image_dicts_list should contain at least two elements.")
    
    num_images = len(image_dicts_list)
    num_rows = (num_images + 1) // 2  # Calculate the required number of rows (rounded up)
    
    # Create a grid of subplots with the appropriate number of rows and 2 columns
    fig, axs = plt.subplots(num_rows, 2, figsize=(width, height * num_rows))
    
    # If the last row has only one image, adjust the subplot layout
    if num_images % 2 == 1:
        fig.delaxes(axs[-1, -1])  # Remove the last subplot
        axs[-1, 0].set_position(axs[-1, -1].get_position())  # Expand the remaining subplot
    
    # Flatten the array of subplots to a 1D array
    axs_flat = axs.flatten()
    
    for i, (ax, image_dict) in enumerate(zip(axs_flat, image_dicts_list)):
        if i < num_images:
            ax.imshow(image_dict["result_plot"])
            ax.set_title(image_dict["model_name"])
            
            # Remove axes and ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.05, wspace=0.05)  # Adjust the vertical and horizontal spacing
    
    # Display the result
    plt.show()


def display_raw_vs_prediction(img_path: str, inferencer: mmdet.models.detectors.BaseDetector, pred_score_thr: float = 0.5, timing: bool = True):
    """Display raw image and prediction visualization side by side.

    Args:
        img_path (str): Path to the input image.
        inferencer (mmdet.models.detectors.BaseDetector): Inference object with detection capabilities.
        pred_score_thr (float, optional): Prediction score threshold. Defaults to 0.5.
        timing (bool, optional): Whether to measure inference timing. Defaults to True.
    """
    pred_dict = generate_pred_plot(img_path=img_path, inferencer=inferencer, pred_score_thr= pred_score_thr, timing=timing)
    # Create a grid of subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    img_raw = mmcv.imread(img_path)
    img_raw = mmcv.imconvert(img_raw, 'bgr', 'rgb')
    axs[0].imshow(img_raw)
    axs[0].set_title(img_path)
    axs[0].axis('off')
    
    axs[1].imshow(pred_dict["result_plot"])
    axs[1].set_title(pred_dict["model_name"])
    axs[1].axis('off')
    
    plt.tight_layout()


def benchmark_model(img_path: str, inferencer_benchmark: List[mmdet.models.detectors.BaseDetector], pred_score_thr: float = 0.5, timing: bool = True):
    """Benchmark multiple detectors by generating and displaying prediction visualizations.

    Args:
        img_path (str): Path to the input image.
        inferencer_benchmark (List[mmdet.models.detectors.BaseDetector]): List of detector inference objects for benchmarking.
        pred_score_thr (float, optional): Prediction score threshold. Defaults to 0.5.
        timing (bool, optional): Whether to measure inference timing. Defaults to True.
    """
    result = []
    for detector in inferencer_benchmark:
        result.append(generate_pred_plot(img_path=img_path, inferencer=detector, pred_score_thr=pred_score_thr, timing=timing))
    display_pred_mosaic(result)
    