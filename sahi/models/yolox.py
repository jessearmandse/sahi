# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_package_minimum_version, check_requirements

from yolox.data.datasets import COCO_CLASSES
from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, postprocess
logger = logging.getLogger(__name__)

class YOLOXDetectionModel(DetectionModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        exp_path: str = None,
        exp_name: str = "yolox-s",
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None
    ):
        self._exp_path = exp_path
        self._exp_name = exp_name
        self._preproc = ValTransform(legacy=False)
        self._nms_threshold = nms_threshold
        super().__init__(
            model_path, 
            model, 
            config_path, 
            device, 
            mask_threshold, 
            confidence_threshold, 
            category_mapping, 
            category_remapping, 
            load_at_init, 
            image_size)

    def check_dependencies(self) -> None:
        check_requirements(["torch", "yolox"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        from yolox.exp import get_exp

        try:
            exp = get_exp(exp_file=self._exp_path, exp_name=self._exp_name)
            self.num_classes = exp.num_classes
            self.nms_threshold = self._nms_threshold if self._nms_threshold < exp.nmsthre else exp.nmsthre

            if self.confidence_threshold < exp.test_conf:
                self.confidence_threshold = exp.test_conf

            if self.image_size is None:
                self.image_size = exp.test_size[0] if exp.test_size[0] > exp.test_size[1] else exp.test_size[1]

            model = exp.get_model()
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid YOLOX model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOX model.
        Args:
            model: Any
                A YOLOX model
        """
        if model.__class__.__module__ not in ["yolox.models.yolox"]:
            raise Exception(f"Not a YOLOX model: {type(model)}")

        if "cuda" in self.device.type:
            model.cuda()
            model.half()
        model.eval()

        if self.model_path is not None:
            ckpt = torch.load(self.model_path, map_location="cpu")
            # load the model state dict, otherwise there'll be no outputs
            model.load_state_dict(ckpt["model"])
            # fuse model by default
            model = fuse_model(model)

        self.model = model

        if not self.category_mapping:
            category_mapping = {ind: category_name for ind, category_name in enumerate(COCO_CLASSES)}
            self.category_mapping = category_mapping
    
    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        image_size = self.image_size

        image_ratio = min(image_size / image.shape[0], image_size / image.shape[1])
        if image_ratio > 1.0:
            preproc_image_size = (image.shape[0], image.shape[1])
            image_ratio = 1.0
        else:
            preproc_image_size = (image_size, image_size)

        logger.info(f"Preprocessing image for prediction, size: {preproc_image_size}")
        img, _ = self._preproc(image, None, preproc_image_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if "cuda" in self.device.type:
            img = img.cuda()
            img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            logger.debug(f"outputs raw {outputs}")
            logger.debug(f"nms threshold {self.nms_threshold}, conf {self.confidence_threshold}")
            outputs = postprocess(
                outputs, self.num_classes, self.confidence_threshold,
                self.nms_threshold, class_agnostic=True
            )
            logger.debug(f"outputs post processed {outputs}")
            outputs = list(filter(lambda o: o is not None, outputs))
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
            predictions = []
            for output in outputs:
                np_output = output.cpu().detach().numpy()
                bboxes = np_output[:, 0:4]
                # preprocessing: resize
                bboxes /= image_ratio

                classes = np_output[:, 6]
                scores = np_output[:, 4] * np_output[:, 5]

                prediction = {
                    "boxes": bboxes,
                    "classes": classes,
                    "scores": scores
                }
                predictions.append(prediction)

            self._original_predictions = predictions
            logger.info(f"Outputs {self.original_predictions}")

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_prediction in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for boxes_ind, prediction in enumerate(image_prediction["boxes"]):
                conf_scores = image_prediction["scores"]
                classes = image_prediction["classes"]
                score = conf_scores[boxes_ind]
                category_id = int(classes[boxes_ind])
                category_name = self.category_mapping[category_id]
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                logger.debug(f'bounding box {bbox}')
                logger.debug(f'score: {score}')
                logger.debug(f'category name: {category_name}')

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)
        
        self._object_prediction_list_per_image = object_prediction_list_per_image if len(object_prediction_list_per_image) > 0 else [[]]
    
    @property
    def original_predictions(self):
        return self._original_predictions
