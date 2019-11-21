from abc import ABC, abstractmethod

from PIL import Image
import sys
sys.path.append('insight_face')
from insight_face.mtcnn import MTCNN
import config
import logging

class FaceCounterBase(ABC):

    @abstractmethod
    def count_faces(self, paths):
        """For the given paths find """
        pass

class MTCNNCounter(FaceCounterBase):

    def count_faces(self, paths):
        """Run MTCNN over the given paths """

        logger = logging.getLogger(config.APP_NAME)

        mtcnn = MTCNN()
        counts = []
        for path in paths:

            image = Image.open(path)
            try:
                bboxes, faces = mtcnn.align_multi(image, config.MAX_FACE_LIMIT, config.MIN_FACE_SIZE)
                counts.append(len(bboxes))
            except ValueError as e:
                logger.warning('For image: %s, encountered expected box stacking issue : %s', path.name, e, exc_info=True)
                counts.append(0)

        return counts

