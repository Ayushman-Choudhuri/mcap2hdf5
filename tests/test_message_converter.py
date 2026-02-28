import struct
from types import SimpleNamespace

import numpy as np
import pytest

from mcap2hdf5.message_converter import MessageConverter


def makePointField(name, offset, datatype):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype)


def makePointCloud2(fields, point_step, points):
    """Build a minimal PointCloud2-like object from a list of per-point byte strings."""
    return SimpleNamespace(
        fields=[makePointField(name, offset, datatype) for name, offset, datatype in fields],
        point_step=point_step,
        data=b"".join(points),
    )


class TestLidarToNumpy:
    def test_output_shape(self):
        """Output shape is (N, len(fieldNames))."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7)]
        points = [struct.pack("<ffff", 1.0, 2.0, 3.0, 0.5)] * 5
        msg = makePointCloud2(fields, point_step=16, points=points)

        result = MessageConverter.lidarToNumpy(msg)

        assert result.shape == (5, 4)

    def test_output_dtype_is_float32(self):
        """Output is always float32 regardless of input field dtypes."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 4)]  # intensity=uint16
        points = [struct.pack("<fffH", 1.0, 2.0, 3.0, 100)]
        msg = makePointCloud2(fields, point_step=14, points=points)

        result = MessageConverter.lidarToNumpy(msg)

        assert result.dtype == np.float32

    def test_packed_float32_fields(self):
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

    def test_fields_with_padding(self):
        """Correctly skips padding bytes between fields (e.g. Velodyne VLP-16 layout).

        Layout: x(0,f32) y(4,f32) z(8,f32) ring(12,u16) pad(14,2B) intensity(16,f32)
        point_step = 20
        """
        fields = [
            ("x", 0, 7),
            ("y", 4, 7),
            ("z", 8, 7),
            ("ring", 12, 4),        # uint16, datatype=4
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

    def test_uint16_field_cast_to_float32(self):
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

    def test_custom_field_names(self):
        """Only the requested fields are extracted; output shape reflects the subset."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7)]
        points = [struct.pack("<ffff", 1.0, 2.0, 3.0, 0.5)]
        msg = makePointCloud2(fields, point_step=16, points=points)

        result = MessageConverter.lidarToNumpy(msg, fieldNames=["x", "y", "z"])

        assert result.shape == (1, 3)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])

    def test_missing_field_raises_value_error(self):
        """Raises ValueError when a requested field is absent from the message."""
        fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7)]
        points = [struct.pack("<fff", 1.0, 2.0, 3.0)]
        msg = makePointCloud2(fields, point_step=12, points=points)

        with pytest.raises(ValueError, match="PointCloud2 missing field: intensity"):
            MessageConverter.lidarToNumpy(msg)
