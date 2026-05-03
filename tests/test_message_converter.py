import struct
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from mcap2hdf5.utils.message_converter import MessageConverter


def makeRawImage(encoding, height, width, data):
    return SimpleNamespace(encoding=encoding, height=height, width=width, data=data)


def makeCompressedImage(format, data):
    return SimpleNamespace(format=format, data=data)


def makePointField(name, offset, datatype):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype)


def makePointCloud2(fields, point_step, points):
    """Build a minimal PointCloud2-like object from a list of per-point byte strings."""
    return SimpleNamespace(
        fields=[makePointField(name, offset, datatype) for name, offset, datatype in fields],
        point_step=point_step,
        data=b"".join(points),
    )


def makeTransformMatrix(translation, euler_xyz_deg):
    """Build a float32 4x4 homogeneous matrix from a translation and XYZ Euler angles (degrees)."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[0:3, 3] = translation
    matrix[0:3, 0:3] = (
        R.from_euler("xyz", euler_xyz_deg, degrees=True).as_matrix().astype(np.float32)
    )
    return matrix


class TestLidarToNumpy:
    def testOutputShape(self):
        """Output shape is (N, len(fieldNames))."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7)]
        points = [struct.pack("<ffff", 1.0, 2.0, 3.0, 0.5)] * 5
        msg = makePointCloud2(fields, point_step=16, points=points)

        result = MessageConverter.lidarToNumpy(msg)

        assert result.shape == (5, 4)

    def testOutputDtypeIsFloat32(self):
        """Output is always float32 regardless of input field dtypes."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 4)]  # intensity=uint16
        points = [struct.pack("<fffH", 1.0, 2.0, 3.0, 100)]
        msg = makePointCloud2(fields, point_step=14, points=points)

        result = MessageConverter.lidarToNumpy(msg)

        assert result.dtype == np.float32

    def testPackedFloat32Fields(self):
        """Correctly reads tightly packed float32 fields with no padding."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7)]
        points = [
            struct.pack("<ffff", 1.0, 2.0, 3.0, 0.5),
            struct.pack("<ffff", 4.0, 5.0, 6.0, 0.8),
        ]
        msg = makePointCloud2(fields, point_step=16, points=points)

        result = MessageConverter.lidarToNumpy(msg)

        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0, 0.5])
        np.testing.assert_array_almost_equal(result[1], [4.0, 5.0, 6.0, 0.8])

    def testFieldsWithPadding(self):
        """Correctly skips padding bytes between fields (e.g. Velodyne VLP-16 layout).

        Layout: x(0,f32) y(4,f32) z(8,f32) ring(12,u16) pad(14,2B) intensity(16,f32)
        point_step = 20
        """
        fields = [
            ("x", 0, 7),
            ("y", 4, 7),
            ("z", 8, 7),
            ("ring", 12, 4),  # uint16, datatype=4
            ("intensity", 16, 7),
        ]
        points = [
            struct.pack("<fffHxxf", 1.0, 2.0, 3.0, 5, 0.5),
            struct.pack("<fffHxxf", 4.0, 5.0, 6.0, 10, 0.8),
        ]
        msg = makePointCloud2(fields, point_step=20, points=points)

        result = MessageConverter.lidarToNumpy(msg)

        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0, 0.5])
        np.testing.assert_array_almost_equal(result[1], [4.0, 5.0, 6.0, 0.8])

    def testUint16FieldCastToFloat32(self):
        """uint16 intensity values are correctly cast to float32."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 4)]  # uint16
        points = [
            struct.pack("<fffH", 1.0, 2.0, 3.0, 100),
            struct.pack("<fffH", 4.0, 5.0, 6.0, 200),
        ]
        msg = makePointCloud2(fields, point_step=14, points=points)

        result = MessageConverter.lidarToNumpy(msg)

        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0, 100.0])
        np.testing.assert_array_almost_equal(result[1], [4.0, 5.0, 6.0, 200.0])

    def testCustomFieldNames(self):
        """Only the requested fields are extracted; output shape reflects the subset."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7)]
        points = [struct.pack("<ffff", 1.0, 2.0, 3.0, 0.5)]
        msg = makePointCloud2(fields, point_step=16, points=points)

        result = MessageConverter.lidarToNumpy(msg, fieldNames=["x", "y", "z"])

        assert result.shape == (1, 3)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])

    def testNoIntensityAliasZeroFills(self):
        """When no intensity alias is present, the 4th column is zero-filled rather than raising."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7)]
        points = [struct.pack("<fff", 1.0, 2.0, 3.0)]
        msg = makePointCloud2(fields, point_step=12, points=points)

        result = MessageConverter.lidarToNumpy(msg)

        assert result.shape == (1, 4)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0, 0.0])

    def testReflectivityAutoDetected(self):
        """Ouster-style 'reflectivity' field is used as the intensity alias."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("reflectivity", 12, 7)]
        points = [struct.pack("<ffff", 1.0, 2.0, 3.0, 42.0)]
        msg = makePointCloud2(fields, point_step=16, points=points)

        result = MessageConverter.lidarToNumpy(msg)

        assert result.shape == (1, 4)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0, 42.0])

    def testExplicitMissingFieldRaisesValueError(self):
        """Explicit fieldNames override still raises if a requested field is absent."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7)]
        points = [struct.pack("<fff", 1.0, 2.0, 3.0)]
        msg = makePointCloud2(fields, point_step=12, points=points)

        with pytest.raises(ValueError, match="PointCloud2 missing field: intensity"):
            MessageConverter.lidarToNumpy(msg, fieldNames=["x", "y", "z", "intensity"])


class TestRawImageToNumpy:
    def testBgr8ReturnsCorrectShape(self):
        """bgr8 raw image is decoded to (H, W, 3) uint8."""
        h, w = 4, 6
        data = bytes(np.zeros((h, w, 3), dtype=np.uint8).tobytes())
        msg = makeRawImage("bgr8", h, w, data)

        result = MessageConverter.rawImageToNumpy(msg)

        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8

    def testRgb8IsConvertedToBgr(self):
        """rgb8 image is converted to BGR — R and B channels are swapped."""
        h, w = 2, 2
        img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        img_rgb[:, :, 0] = 255  # R channel
        msg = makeRawImage("rgb8", h, w, bytes(img_rgb.tobytes()))

        result = MessageConverter.rawImageToNumpy(msg)

        assert result.shape == (h, w, 3)
        assert result[0, 0, 2] == 255  # R moved to BGR[2]
        assert result[0, 0, 0] == 0    # B channel is zero

    def testBgra8IsConvertedToBgr(self):
        """bgra8 alpha channel is dropped; output is (H, W, 3)."""
        h, w = 2, 2
        data = bytes(np.ones((h, w, 4), dtype=np.uint8).tobytes())
        msg = makeRawImage("bgra8", h, w, data)

        result = MessageConverter.rawImageToNumpy(msg)

        assert result.shape == (h, w, 3)

    def testUnsupportedEncodingRaisesValueError(self):
        """Unsupported encodings raise ValueError with a helpful message."""
        msg = makeRawImage("mono8", 4, 4, bytes(16))

        with pytest.raises(ValueError, match="Unsupported raw image encoding 'mono8'"):
            MessageConverter.rawImageToNumpy(msg)

    def testImageToNumpyDispatchesOnFormat(self):
        """imageToNumpy routes to compressedImageToNumpy when message has a 'format' field."""
        import cv2
        h, w = 8, 8
        img = np.zeros((h, w, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        msg = makeCompressedImage("jpeg", buf.tobytes())

        result = MessageConverter.imageToNumpy(msg)

        assert result.shape == (h, w, 3)

    def testImageToNumpyDispatchesOnEncoding(self):
        """imageToNumpy routes to rawImageToNumpy when message has no 'format' field."""
        h, w = 4, 4
        data = bytes(np.zeros((h, w, 3), dtype=np.uint8).tobytes())
        msg = makeRawImage("bgr8", h, w, data)

        result = MessageConverter.imageToNumpy(msg)

        assert result.shape == (h, w, 3)


class TestInterpolateMatrix:
    def testAlphaZeroReturnsStartMatrix(self):
        """alpha=0 returns startMatrix exactly."""
        start = makeTransformMatrix([1, 2, 3], [0, 0, 0])
        end = makeTransformMatrix([4, 5, 6], [0, 0, 90])

        result = MessageConverter.interpolateMatrix(start, end, alpha=0.0)

        np.testing.assert_array_almost_equal(result, start)

    def testAlphaOneReturnsEndMatrix(self):
        """alpha=1 returns endMatrix exactly."""
        start = makeTransformMatrix([1, 2, 3], [0, 0, 0])
        end = makeTransformMatrix([4, 5, 6], [0, 0, 90])

        result = MessageConverter.interpolateMatrix(start, end, alpha=1.0)

        np.testing.assert_array_almost_equal(result, end)

    def testDefaultAlphaMatchesExplicitHalf(self):
        """Omitting alpha produces the same result as passing alpha=0.5."""
        start = makeTransformMatrix([0, 0, 0], [0, 0, 0])
        end = makeTransformMatrix([2, 4, 6], [0, 0, 60])

        result_default = MessageConverter.interpolateMatrix(start, end)
        result_explicit = MessageConverter.interpolateMatrix(start, end, alpha=0.5)

        np.testing.assert_array_equal(result_default, result_explicit)

    def testTranslationIsLinearlyInterpolated(self):
        """Translation is lerped independently of rotation."""
        start = makeTransformMatrix([0, 0, 0], [0, 0, 0])
        end = makeTransformMatrix([4, 8, 12], [0, 0, 0])

        result = MessageConverter.interpolateMatrix(start, end, alpha=0.25)

        np.testing.assert_array_almost_equal(result[0:3, 3], [1.0, 2.0, 3.0])

    def testRotationMidpointIsCorrect(self):
        """Rotation at alpha=0.5 is the exact angular midpoint (SLERP, not NLERP)."""
        start = makeTransformMatrix([0, 0, 0], [0, 0, 0])
        end = makeTransformMatrix([0, 0, 0], [0, 0, 90])

        result = MessageConverter.interpolateMatrix(start, end, alpha=0.5)

        expected_rot = R.from_euler("z", 45, degrees=True).as_matrix().astype(np.float32)
        np.testing.assert_array_almost_equal(result[0:3, 0:3], expected_rot)

    def testRotationBlockIsOrthonormal(self):
        """The 3x3 rotation block satisfies R^T R = I and det(R) = 1."""
        start = makeTransformMatrix([0, 0, 0], [10, 20, 30])
        end = makeTransformMatrix([1, 2, 3], [40, 50, 60])

        result = MessageConverter.interpolateMatrix(start, end, alpha=0.3)

        rot = result[0:3, 0:3].astype(np.float64)
        np.testing.assert_array_almost_equal(rot @ rot.T, np.eye(3))
        assert abs(np.linalg.det(rot) - 1.0) < 1e-6

    def testOutputShapeAndBottomRow(self):
        """Output is (4, 4) and the bottom row is always [0, 0, 0, 1]."""
        start = makeTransformMatrix([1, 0, 0], [0, 0, 0])
        end = makeTransformMatrix([0, 1, 0], [0, 0, 45])

        result = MessageConverter.interpolateMatrix(start, end, alpha=0.5)

        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result[3], [0, 0, 0, 1])

    def testOutputDtypeIsFloat32(self):
        """Output matrix is always float32."""
        start = makeTransformMatrix([0, 0, 0], [0, 0, 0])
        end = makeTransformMatrix([1, 1, 1], [0, 0, 90])

        result = MessageConverter.interpolateMatrix(start, end, alpha=0.5)

        assert result.dtype == np.float32

    def testLargeRotationTakesShortestPath(self):
        """SLERP takes the shortest arc; 270° rotation interpolates as -90°."""
        start = makeTransformMatrix([0, 0, 0], [0, 0, 0])
        # 270° around z — shortest path is -90°, so midpoint should be -45° (315°)
        end = makeTransformMatrix([0, 0, 0], [0, 0, 270])

        result = MessageConverter.interpolateMatrix(start, end, alpha=0.5)

        # scipy Slerp takes the short way: midpoint is at -45° (i.e. 315°)
        expected_rot = R.from_euler("z", -45, degrees=True).as_matrix().astype(np.float32)
        np.testing.assert_array_almost_equal(result[0:3, 0:3], expected_rot)
