OCR_CONFIG = { 
    "det": './configs/textdet/dbnet/dbnet_resnet50-oclip_1200e_icdar2015.py', 
    "det_weights": './tiresias/model/ocr/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth',
    "rec": './configs/textrecog/svtr/svtr-small_20e_st_mj.py',
    "rec_weights": './tiresias/model/ocr/svtr-small_20e_st_mj-35d800d6.pth'
}

OCR_ALLOW_INPUT = {".jpg", ".JPG", ".jpeg", ".JPEG"}

OCR_DISPLAY_VAR = {
    'neighbor_color': 'red',
    'ocr_color': 'yellow',
    'labo_color': 'blue',
    'column_id': 'GALERIE'
}
