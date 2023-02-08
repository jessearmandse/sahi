import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.yolox import YOLOXTestConstants, download_yolox_s_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 320

class TestYOLOXDetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.models.yolox import YOLOXDetectionModel

        download_yolox_s_model()

        yolox_detection_model = YOLOXDetectionModel(
            model_path=YOLOXTestConstants.YOLOX_S_MODEL_PATH,
            exp_path=YOLOXTestConstants.YOLOX_S_EXP_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE
        )

        self.assertNotEqual(yolox_detection_model, None)

    def test_set_model(self):
        from yolox.exp import get_exp
        from sahi.models.yolox import YOLOXDetectionModel

        download_yolox_s_model()

        exp = get_exp(exp_name="yolox_s")
        yolo_model = exp.get_model()
        yolox_detection_model = YOLOXDetectionModel(
            model=yolo_model,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE
        )

        self.assertNotEqual(yolox_detection_model, None)
    
    def test_perform_inference(self):
        from sahi.models.yolox import YOLOXDetectionModel
        from yolox.exp import get_exp

        download_yolox_s_model()

        yolox_detection_model = YOLOXDetectionModel(
            model_path=YOLOXTestConstants.YOLOX_S_MODEL_PATH,
            exp_path=YOLOXTestConstants.YOLOX_S_EXP_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        yolox_detection_model.perform_inference(image)
        original_predictions = yolox_detection_model.original_predictions
        boxes = original_predictions[0]["boxes"]
        classes = original_predictions[0]["classes"]
        conf = original_predictions[0]["scores"]
        
        # compare
        desired_bbox = [321, 329, 378, 368]
        within_desired_bbox = False
        for ind, box in enumerate(boxes):
            if classes[ind].item() == 2:
                self.assertGreater(conf[ind].item(), 0.5)
            predicted_bbox = list(map(int, box[:4].tolist()))
            margin = 7
            print(f'desired box, {desired_bbox}, predicted box: {predicted_bbox} within margin {margin}')

            for ind, point in enumerate(predicted_bbox):
                if point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin:
                    within_desired_bbox = True
        self.assertTrue(within_desired_bbox)

        self.assertEqual(yolox_detection_model.num_classes, 80)
        for ind, box in enumerate(boxes):
            self.assertGreaterEqual(conf[ind].item(), CONFIDENCE_THRESHOLD)

    def test_convert_original_predictions(self):
        from sahi.models.yolox import YOLOXDetectionModel
        from yolox.exp import get_exp

        download_yolox_s_model()

        yolox_detection_model = YOLOXDetectionModel(
            model_path=YOLOXTestConstants.YOLOX_S_MODEL_PATH,
            exp_path=YOLOXTestConstants.YOLOX_S_EXP_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        yolox_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        yolox_detection_model.convert_original_predictions()
        object_prediction_list = yolox_detection_model.object_prediction_list
        print(f'object prediction list {object_prediction_list}')

        self.assertEqual(len(object_prediction_list), 3)
        desired_bbox = [321, 329, 57, 39]
        within_desired_bbox = False

        for object_prediction in object_prediction_list:
            self.assertEqual(object_prediction.category.id, 2)
            self.assertEqual(object_prediction.category.name, "car")

            predicted_bbox = object_prediction.bbox.to_xywh()
            margin = 9
            print(f'desired box, {desired_bbox}, predicted box: {predicted_bbox} within margin {margin}')

            for ind, point in enumerate(predicted_bbox):
                if point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin:
                    within_desired_bbox = True

            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)
        
        self.assertTrue(within_desired_bbox)

    
if __name__ == "__main__":
    unittest.main()