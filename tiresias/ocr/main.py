# new dom
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import tiresias.utils.geo_info
from tiresias.config import OCR_ALLOW_INPUT, OCR_CONFIG
import tiresias.utils.data
import tiresias.ocr.ocr_infer
import tiresias.ocr.ocr_viz
from tiresias.data import LABO_GALERIE_SHAPEFILE

import matplotlib.pyplot as plt
from typing import Optional

COLUMN_ID_GALERIE = LABO_GALERIE_SHAPEFILE['column_id']
#TODO add cintre
#COLUMN_ID_CINTRE = 


def get_galerie_ocr(ocr_pred_df: pd.DataFrame, galerie_name: pd.Series) -> Optional[pd.DataFrame]:
    """ Return ocr prediction filtered on possible galerie name.

    Args:
        ocr_prediction (pd.DataFrame): _description_
        galerie_name (pd.Series): _description_

    Returns:
        _type_: _description_
    """
    ocr_pred_df['rec_texts'] = ocr_pred_df['rec_texts'].apply(lambda x : x.upper())
    galerie_prediction_df = ocr_pred_df[ocr_pred_df['rec_texts'].isin(galerie_name)]
    if galerie_prediction_df.shape[0] == 0:
        return None
    return galerie_prediction_df


def test_galerie_unique(ocr_pred_galerie_df: pd.DataFrame) -> bool:
    """ _summary_

    Args:
        ocr_pred_galerie (_type_): _description_

    Returns:
        _type_: _description_
    """
    return ocr_pred_galerie_df.rec_texts.unique().size == 1


#OCR_DOM = tiresias.ocr.ocr_infer.load_ocr_inferencer(device='cuda')


def main(image_path: str, device: str ='cpu', ocr_config: dict = OCR_CONFIG):
    # load data path
    paths = tiresias.utils.data.get_path_from_input(input_path=image_path, allowed_extensions=OCR_ALLOW_INPUT)
    _, galerie_gdf = tiresias.utils.geo_info.load_shapefile()
    print(f"start ocr inference on {image_path}")
    # load pretrained ocr
    det = ocr_config['det']
    det_weights = ocr_config['det_weights']
    rec = ocr_config['rec']
    rec_weights = ocr_config['rec_weights']
    ocr = tiresias.ocr.ocr_infer.load_ocr_inferencer(det=det, det_weights=det_weights, rec=rec, rec_weights=rec_weights, device=device)
    # inference
    ocr_pred = tiresias.ocr.ocr_infer.infer_ocr(image_path=image_path, mmocr=ocr)
    # case no detection
    if ocr_pred is None:
        print("No text detected")
        return
    # otherwise transform result in usable DF
    ocr_pred_df = tiresias.ocr.ocr_infer.ocr_predictions_to_df(ocr_pred)
    # check if specific detection
    ocr_pred_galerie_df = get_galerie_ocr(ocr_pred_df=ocr_pred_df, galerie_name=galerie_gdf[COLUMN_ID_GALERIE])
    if ocr_pred_galerie_df is None:
        print("No galerie detected")
        plt.title(f'File : {image_path}')
        plt.imshow(Image.open(image_path))
        return
    if test_galerie_unique(ocr_pred_galerie_df=ocr_pred_galerie_df):
        detected_galerie_name = ocr_pred_galerie_df.rec_texts.unique()[0]
        tiresias.ocr.ocr_viz.plot_ocr_results(image_path=image_path, ocr_pred_galerie_df=ocr_pred_galerie_df, galerie_gdf=galerie_gdf, detected_galerie_name=detected_galerie_name)
    # # check if specific detection
    # ocr_pred_cintre = None
    # if ocr_pred_cintre is None and ocr_pred_galerie.rec_texts.unique().size == 1:
    #     print("No cintre detected")
    #     localize.display_id(gdf=galerie, id_value='GAN')
    return

if __name__ == "main":
    print('coucou')
    main('/home/dominique/mmocr/data_andra/img/ocr/NED_06-10-2016 (13).JPG')
    # img_path = pathlib.Path('.').glob('./data_andra/img/ocr/*')
    # for i in range(5):
    #     img = str(img_path.__next__())
    #     main(img)
    #     break
