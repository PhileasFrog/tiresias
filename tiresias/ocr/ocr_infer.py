import pandas as pd
import time
import tiresias.ocr.ocr_viz
from mmocr.apis import MMOCRInferencer
from tiresias.config import OCR_CONFIG
from typing import Dict, Optional, List



def load_ocr_inferencer(
    det: str = OCR_CONFIG['det'],
    det_weights: str = OCR_CONFIG['det_weights'],
    rec: str = OCR_CONFIG['rec'],
    rec_weights: str = OCR_CONFIG['rec_weights'],
    device: Optional[str] = "cpu"
) -> MMOCRInferencer:
    """
    Load an Optical Character Recognition (OCR) model inference object.

    This function initializes and returns an instance of MMOCRInferencer,
    which is capable of performing OCR tasks using the specified detection and
    recognition models.

    Args:
        det (str, optional): Path to the detection model configuration file.
            Defaults to OCR_CONFIG['det'].
        det_weights (str, optional): Path to the detection model weights file.
            Defaults to OCR_CONFIG['det_weights'].
        rec (str, optional): Path to the recognition model configuration file.
            Defaults to OCR_CONFIG['rec'].
        rec_weights (str, optional): Path to the recognition model weights file.
            Defaults to OCR_CONFIG['rec_weights'].
        device (str, optional): Device to run inference on (e.g., 'cpu' or 'cuda').
            Defaults to 'cpu'.

    Returns:
        MMOCRInferencer: An instance of MMOCRInferencer with the loaded OCR models.

    Note:
        MMOCRInferencer is a custom class that should be imported from the appropriate module.
        OCR_CONFIG is a dictionary containing the default paths for OCR configurations and weights.
    """
    return MMOCRInferencer(
        det=det,
        det_weights=det_weights,
        rec=rec,
        rec_weights=rec_weights,
        device=device
    )


def infer_ocr(image_path: str, mmocr: MMOCRInferencer) -> Optional[Dict[str, str]]:
    """
    Perform Optical Character Recognition (OCR) on the specified image.

    This function uses the provided MMOCRInferencer object to perform OCR
    on the image located at the given image_path.

    Args:
        image_path (str): Path to the image file for OCR.
        mmocr (MMOCRInferencer): An instance of MMOCRInferencer with loaded OCR models.

    Returns:
        Dict[str, str]: A dictionary containing the OCR predictions.
            The keys represent the predictions and the values are the corresponding vizualisations.
    """
    predictions = mmocr(inputs=image_path)
    if not predictions['predictions'][0]['rec_texts']:
        print("No text detected")
        return None
    return predictions



def ocr_predictions_to_df(ocr_prediction: Dict[str, any]) -> Optional[pd.DataFrame]:
    """
    Convert OCR predictions to a pandas DataFrame.

    This function takes OCR predictions in the form of a dictionary and converts
    them to a pandas DataFrame. The input `ocr_prediction` is assumed to have a
    structure similar to the output of the `infer_ocr` function, where OCR
    predictions are stored under the 'predictions' key.

    Args:
        ocr_prediction (Dict[str, any]): OCR predictions, typically obtained from `infer_ocr`.

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame containing the OCR predictions.
            The DataFrame will have columns representing various OCR attributes
            such as 'text', 'confidence', 'box', etc.

            If the OCR predictions are empty (i.e., no text detected), the function
            returns None.

    Note:
        The structure of the `ocr_prediction` dictionary may vary based on the OCR model used.
        Please ensure that the `ocr_prediction` dictionary contains OCR predictions in the
        expected format to avoid errors during DataFrame creation.
    """
    dfs = []
    for prediction_data in ocr_prediction['predictions']:
        df = pd.DataFrame(prediction_data)
        dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df


def test_perf_gpu_cpu(image_path: str, n_iter: int = 3):
    """
    Compare the performance of OCR inference on GPU and CPU for multiple iterations.

    Parameters:
    image_path (str): Path to the input image.
    n_iter (int, optional): Number of iterations for testing. Default is 3.

    Returns:
    None
    """
    time_gpu = []
    time_cpu = []

    # Load OCR inference models for CPU and GPU
    ocr_cpu = load_ocr_inferencer(device='cpu')
    ocr_gpu = load_ocr_inferencer(device='cuda')

    for i in range(n_iter):
        # Measure GPU inference time
        a = time.time()
        ocr_gpu(image_path)
        b = time.time()
        print(f"GPU inference iteration {i} took {b - a} seconds")
        time_gpu.append(b - a)

        # Measure CPU inference time
        a = time.time()
        ocr_cpu(image_path)
        b = time.time()
        print(f"CPU inference iteration {i} took {b - a} seconds")
        time_cpu.append(b - a)

    # Calculate and display average inference times
    avg_time_gpu = sum(time_gpu) / len(time_gpu)
    avg_time_cpu = sum(time_cpu) / len(time_cpu)
    print(f"Average GPU inference time: {avg_time_gpu} seconds")
    print(f"Average CPU inference time: {avg_time_cpu} seconds")


def infer_time_plot(image_path: str, ocr_infer: MMOCRInferencer, output_text: bool = False) -> Optional[List[str]]:
    """
    Perform OCR inference on an image, report the inference time, and display raw image alongside the image with detections.

    Parameters:
    image_path (str): Path to the input image.
    ocr_infer (MMOCRInferencer): An instance of MMOCRInferencer with loaded OCR models.
    output_text (bool, optional): Whether to output the recognized text. Default is False.

    Returns:
    Optional[List[str]]: A list of recognized text strings if output_text is True, otherwise None.
    """
    a = time.time()
    ocr_pred = infer_ocr(image_path=image_path, mmocr=ocr_infer)
    if ocr_pred is None:
        return
    
    b = time.time()
    print(f"Inference took {b - a} seconds")
    
    ocr_pred_df = ocr_predictions_to_df(ocr_pred)
    tiresias.ocr.ocr_viz.plot_ocr_results_raw(image_path=image_path, ocr_pred_df=ocr_pred_df)
    
    if output_text:
        rec_texts = [rec_text["rec_texts"] for rec_text in ocr_pred['predictions']][0][::-1]
        return rec_texts