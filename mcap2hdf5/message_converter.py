import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

DEFAULT_LIDAR_MESSAGE_FIELDS = ["x", "y", "z", "intensity"]

POINTFIELD_DATATYPE_MAP = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}

class MessageConverter:
    @staticmethod
    def lidarToNumpy(lidarMsg, fieldNames=DEFAULT_LIDAR_MESSAGE_FIELDS):
        fieldMap = {field.name: field for field in lidarMsg.fields}

        for fieldName in fieldNames:
            if fieldName not in fieldMap:
                raise ValueError(f"PointCloud2 missing field: {fieldName}")

        pointByteBuffer = np.frombuffer(lidarMsg.data, dtype=np.uint8)
        pointByteBuffer = pointByteBuffer.reshape(-1, lidarMsg.point_step)
        pointCloud = np.empty((len(pointByteBuffer), len(fieldNames)), dtype=np.float32)

        for index, fieldName in enumerate(fieldNames):
            field = fieldMap[fieldName]
            fieldDtype = np.dtype(POINTFIELD_DATATYPE_MAP[field.datatype])
            fieldEndByte = field.offset + fieldDtype.itemsize
            fieldBytes = pointByteBuffer[:, field.offset:fieldEndByte].tobytes()
            pointCloud[:, index] = np.frombuffer(fieldBytes, dtype=fieldDtype)

        return pointCloud

    @staticmethod
    def compressedImageToNumpy(imageMsg):
        imageArray = np.frombuffer(imageMsg.data, np.uint8)
        cv2Image = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)
        return cv2Image

    @staticmethod
    def transformToMatrix(transform):
        translation = transform.translation
        rotation = transform.rotation
        matrix = np.eye(4, dtype=np.float32)
        matrix[0:3, 3] = [
            translation.x,
            translation.y,
            translation.z,
        ]
        matrix[0:3, 0:3] = R.from_quat([
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        ]).as_matrix().astype(np.float32)
        return matrix

    @staticmethod
    def interpolateMatrix(startMatrix, endMatrix, alpha=0.5):
        interpolatedMatrix = np.eye(4, dtype=np.float32)

        interpolatedMatrix[0:3, 3] = (1.0 - alpha) * startMatrix[0:3, 3] + alpha * endMatrix[0:3, 3]

        rotations = R.concatenate([
            R.from_matrix(startMatrix[0:3, 0:3]),
            R.from_matrix(endMatrix[0:3, 0:3]),
        ])
        interpolatedMatrix[0:3, 0:3] = (
            Slerp([0, 1], rotations)(alpha).as_matrix().astype(np.float32)
        )

        return interpolatedMatrix
