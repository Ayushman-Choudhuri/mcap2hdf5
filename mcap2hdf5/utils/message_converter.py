import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

INTENSITY_FIELD_ALIASES = ["intensity", "reflectivity", "signal"]
SPATIAL_FIELDS = ["x", "y", "z"]

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

SUPPORTED_RAW_ENCODINGS = {"rgb8", "bgr8", "rgba8", "bgra8"}

RAW_TO_BGR = {
    "rgb8": cv2.COLOR_RGB2BGR,
    "rgba8": cv2.COLOR_RGBA2BGR,
    "bgra8": cv2.COLOR_BGRA2BGR,
}


class MessageConverter:
    @staticmethod
    def lidarToNumpy(lidarMsg, fieldNames=None):
        fieldMap = {field.name: field for field in lidarMsg.fields}

        if fieldNames is None:
            for name in SPATIAL_FIELDS:
                if name not in fieldMap:
                    raise ValueError(f"PointCloud2 missing required spatial field: {name}")

            intensityField = next((f for f in INTENSITY_FIELD_ALIASES if f in fieldMap), None)
            resolvedFields = SPATIAL_FIELDS + ([intensityField] if intensityField else [])
            outputCols = 4
        else:
            for name in fieldNames:
                if name not in fieldMap:
                    raise ValueError(f"PointCloud2 missing field: {name}")
            resolvedFields = fieldNames
            outputCols = len(fieldNames)

        pointByteBuffer = np.frombuffer(lidarMsg.data, dtype=np.uint8)
        pointByteBuffer = pointByteBuffer.reshape(-1, lidarMsg.point_step)
        pointCloud = np.zeros((len(pointByteBuffer), outputCols), dtype=np.float32)

        for index, fieldName in enumerate(resolvedFields):
            field = fieldMap[fieldName]
            fieldDtype = np.dtype(POINTFIELD_DATATYPE_MAP[field.datatype])
            fieldEndByte = field.offset + fieldDtype.itemsize
            fieldBytes = pointByteBuffer[:, field.offset : fieldEndByte].tobytes()
            pointCloud[:, index] = np.frombuffer(fieldBytes, dtype=fieldDtype)

        return pointCloud

    @staticmethod
    def compressedImageToNumpy(imageMsg):
        imageArray = np.frombuffer(imageMsg.data, np.uint8)
        return cv2.imdecode(imageArray, cv2.IMREAD_COLOR)

    @staticmethod
    def rawImageToNumpy(imageMsg):
        encoding = imageMsg.encoding.lower()
        if encoding not in SUPPORTED_RAW_ENCODINGS:
            raise ValueError(
                f"Unsupported raw image encoding '{imageMsg.encoding}'. "
                f"Supported: {sorted(SUPPORTED_RAW_ENCODINGS)}"
            )
        channels = 4 if "a" in encoding else 3
        img = np.frombuffer(imageMsg.data, dtype=np.uint8).reshape(
            imageMsg.height, imageMsg.width, channels
        )
        if encoding in RAW_TO_BGR:
            img = cv2.cvtColor(img, RAW_TO_BGR[encoding])
        return img

    @staticmethod
    def imageToNumpy(imageMsg):
        """Dispatch to compressed or raw decoder based on message type."""
        if hasattr(imageMsg, "format"):
            return MessageConverter.compressedImageToNumpy(imageMsg)
        return MessageConverter.rawImageToNumpy(imageMsg)

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
        matrix[0:3, 0:3] = (
            R.from_quat(
                [
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                    float(rotation.w),
                ]
            )
            .as_matrix()
            .astype(np.float32)
        )
        return matrix

    @staticmethod
    def interpolateMatrix(startMatrix, endMatrix, alpha=0.5):
        interpolatedMatrix = np.eye(4, dtype=np.float32)

        interpolatedMatrix[0:3, 3] = (1.0 - alpha) * startMatrix[0:3, 3] + alpha * endMatrix[0:3, 3]

        rotations = R.concatenate(
            [
                R.from_matrix(startMatrix[0:3, 0:3]),
                R.from_matrix(endMatrix[0:3, 0:3]),
            ]
        )
        interpolatedMatrix[0:3, 0:3] = (
            Slerp([0, 1], rotations)(alpha).as_matrix().astype(np.float32)
        )

        return interpolatedMatrix
