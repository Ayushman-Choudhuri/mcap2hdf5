import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

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
            end = field.offset + fieldDtype.itemsize
            fieldBytes = pointByteBuffer[:, field.offset:end].tobytes()
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
    def interpolateMatrix(matrix1, matrix2, alpha):
        result = np.eye(4, dtype=np.float32)

        translation1 = matrix1[0:3, 3]
        translation2 = matrix2[0:3, 3]
        result[0:3, 3] = (1.0 - alpha) * translation1 + alpha * translation2

        rotation1 = R.from_matrix(matrix1[0:3, 0:3])
        rotation2 = R.from_matrix(matrix2[0:3, 0:3])

        quat1 = rotation1.as_quat()
        quat2 = rotation2.as_quat()

        if np.dot(quat1, quat2) < 0:
            quat2 = -quat2

        quatInterp = (1.0 - alpha) * quat1 + alpha * quat2
        quatInterp = quatInterp / np.linalg.norm(quatInterp)

        result[0:3, 0:3] = R.from_quat(quatInterp).as_matrix().astype(np.float32)

        return result
