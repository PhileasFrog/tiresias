import time
import mmdet.models.detectors
import mmdet.structures.det_data_sample
from mmdet.apis import init_detector, inference_detector


def load_inference_detector(config_file: str, checkpoint_file: str, device: str) -> mmdet.models.detectors.BaseDetector:
    """Load an inference detector model.

    Args:
        config_file (str): Path to the model configuration file.
        checkpoint_file (str): Path to the model checkpoint file.
        device (str): Device for inference (e.g., 'cuda:0' or 'cpu').

    Returns:
        mmdet.models.detectors.BaseDetector: Inference detector model.
    """
    model = init_detector(config_file, checkpoint_file, device=device)
    return model


def infer_detection(img_path: str, inferencer: mmdet.models.detectors.BaseDetector, timing: bool = False) -> mmdet.structures.det_data_sample.DetDataSample:
    """Perform object detection inference on an image using the given detector.

    Args:
        img_path (str): Path to the input image.
        inferencer (mmdet.models.detectors.BaseDetector): Inference detector object.
        timing (bool, optional): Whether to measure inference timing. Defaults to False.

    Returns:
        DetDataSample: Detection results.
    """
    if timing:
        a =time.time()
    result = inference_detector(inferencer, img_path)
    if timing:
        b = time.time()
        try:
            if inferencer.custom:
                model_name = 'Custom_' + inferencer._get_name()
        except AttributeError:
            model_name = inferencer._get_name()
        print(f"Time to infer: {b-a} seconds on {inferencer.data_preprocessor.device.type} for model {model_name}")
    return result


