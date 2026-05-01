"""Tests for cli.py and cli_utils.py."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mcap2hdf5.cli import app
from mcap2hdf5.utils.cli_utils import DetectedSensors, detectFirst, detectSensors, detectTF

runner = CliRunner()


class TestDetectFirst:
    def testReturnsFirstMatchingTopic(self):
        topicToSchema = {"/camera/image": "sensor_msgs/msg/Image"}
        result = detectFirst(topicToSchema, {"sensor_msgs/msg/Image"}, "camera image")
        assert result == "/camera/image"

    def testReturnsNoneWhenNoMatch(self):
        topicToSchema = {"/camera/image": "sensor_msgs/msg/Image"}
        result = detectFirst(topicToSchema, {"sensor_msgs/msg/PointCloud2"}, "lidar")
        assert result is None

    def testReturnsFirstWhenMultipleMatch(self):
        topicToSchema = {
            "/cam0/image": "sensor_msgs/msg/Image",
            "/cam1/image": "sensor_msgs/msg/Image",
        }
        result = detectFirst(topicToSchema, {"sensor_msgs/msg/Image"}, "camera image")
        assert result == "/cam0/image"

    def testEmptyTopicMap(self):
        result = detectFirst({}, {"sensor_msgs/msg/Image"}, "camera image")
        assert result is None

    def testMatchesOnlyExactSchemaName(self):
        topicToSchema = {"/lidar": "sensor_msgs/msg/PointCloud2Extra"}
        result = detectFirst(topicToSchema, {"sensor_msgs/msg/PointCloud2"}, "lidar")
        assert result is None


class TestDetectTF:
    def testDetectsDynamicAndStaticTF(self):
        topicToSchema = {
            "/tf": "tf2_msgs/msg/TFMessage",
            "/tf_static": "tf2_msgs/msg/TFMessage",
        }
        dynamic, static = detectTF(topicToSchema)
        assert dynamic == "/tf"
        assert static == "/tf_static"

    def testReturnsNoneWhenBothAbsent(self):
        dynamic, static = detectTF({})
        assert dynamic is None
        assert static is None

    def testReturnsDynamicOnlyWhenNoStatic(self):
        topicToSchema = {"/tf": "tf2_msgs/msg/TFMessage"}
        dynamic, static = detectTF(topicToSchema)
        assert dynamic == "/tf"
        assert static is None

    def testReturnsStaticOnlyWhenNoDynamic(self):
        topicToSchema = {"/tf_static": "tf2_msgs/msg/TFMessage"}
        dynamic, static = detectTF(topicToSchema)
        assert dynamic is None
        assert static == "/tf_static"

    def testIgnoresNonTFSchemas(self):
        topicToSchema = {"/tf": "sensor_msgs/msg/Image"}
        dynamic, static = detectTF(topicToSchema)
        assert dynamic is None
        assert static is None

    def testStaticDetectionIsCaseInsensitive(self):
        topicToSchema = {
            "/TF_STATIC": "tf2_msgs/msg/TFMessage",
            "/tf": "tf2_msgs/msg/TFMessage",
        }
        dynamic, static = detectTF(topicToSchema)
        assert static == "/TF_STATIC"
        assert dynamic == "/tf"


class TestDetectSensors:
    def testFullDetection(self):
        topicToSchema = {
            "/camera/image_raw": "sensor_msgs/msg/Image",
            "/camera/camera_info": "sensor_msgs/msg/CameraInfo",
            "/velodyne_points": "sensor_msgs/msg/PointCloud2",
            "/tf": "tf2_msgs/msg/TFMessage",
            "/tf_static": "tf2_msgs/msg/TFMessage",
        }
        result = detectSensors(topicToSchema)
        assert result == DetectedSensors(
            cameraImage="/camera/image_raw",
            cameraInfo="/camera/camera_info",
            lidar="/velodyne_points",
            tf="/tf",
            tfStatic="/tf_static",
        )

    def testAllNoneWhenEmpty(self):
        result = detectSensors({})
        assert result == DetectedSensors(None, None, None, None, None)

    def testCompressedImageDetected(self):
        topicToSchema = {"/camera/compressed": "sensor_msgs/msg/CompressedImage"}
        result = detectSensors(topicToSchema)
        assert result.cameraImage == "/camera/compressed"

    def testReturnTypeIsDetectedSensors(self):
        result = detectSensors({})
        assert isinstance(result, DetectedSensors)

    def testSupportsUnpacking(self):
        topicToSchema = {"/camera/image_raw": "sensor_msgs/msg/Image"}
        camImg, camInfo, lidar, tf, tfStatic = detectSensors(topicToSchema)
        assert camImg == "/camera/image_raw"
        assert camInfo is None
        assert lidar is None
        assert tf is None
        assert tfStatic is None


def _makeChannelSummary(topics: dict[str, str]) -> MagicMock:
    """Build a minimal MCAP summary mock with the given {topic: schema_name} mapping."""
    channels = {}
    schemas = {}
    channel_message_counts = {}
    for idx, (topic, schema_name) in enumerate(topics.items(), start=1):
        channel = SimpleNamespace(id=idx, topic=topic, schema_id=idx)
        schema = SimpleNamespace(id=idx, name=schema_name)
        channels[idx] = channel
        schemas[idx] = schema
        channel_message_counts[idx] = 10

    statistics = SimpleNamespace(channel_message_counts=channel_message_counts)
    summary = MagicMock()
    summary.channels = channels
    summary.schemas = schemas
    summary.statistics = statistics
    return summary


_FULL_TOPICS = {
    "/camera/image_raw": "sensor_msgs/msg/Image",
    "/camera/camera_info": "sensor_msgs/msg/CameraInfo",
    "/velodyne_points": "sensor_msgs/msg/PointCloud2",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
}


class TestCliInspect:
    def testInspectCallsRunInspect(self, tmp_path):
        mcap = tmp_path / "test.mcap"
        mcap.write_bytes(b"")
        summary = _makeChannelSummary(_FULL_TOPICS)

        with patch("mcap2hdf5.utils.cli_utils.make_reader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_summary.return_value = summary
            mock_reader_cls.return_value = mock_reader

            result = runner.invoke(app, ["inspect", str(mcap)])

        assert result.exit_code == 0

    def testInspectPrintsTopicsInOutput(self, tmp_path):
        mcap = tmp_path / "test.mcap"
        mcap.write_bytes(b"")
        summary = _makeChannelSummary({"/velodyne_points": "sensor_msgs/msg/PointCloud2"})

        with patch("mcap2hdf5.utils.cli_utils.make_reader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_summary.return_value = summary
            mock_reader_cls.return_value = mock_reader

            result = runner.invoke(app, ["inspect", str(mcap)])

        assert "/velodyne_points" in result.output

    def testInspectFileNotFound(self):
        result = runner.invoke(app, ["inspect", "/nonexistent/path/file.mcap"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def testInspectPrintsAutoDetection(self, tmp_path):
        mcap = tmp_path / "test.mcap"
        mcap.write_bytes(b"")
        summary = _makeChannelSummary(_FULL_TOPICS)

        with patch("mcap2hdf5.utils.cli_utils.make_reader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_summary.return_value = summary
            mock_reader_cls.return_value = mock_reader

            result = runner.invoke(app, ["inspect", str(mcap)])

        assert "Auto-detection" in result.output

    def testInspectShowsNotFoundForMissingModality(self, tmp_path):
        mcap = tmp_path / "test.mcap"
        mcap.write_bytes(b"")
        # only lidar, no camera or TF
        summary = _makeChannelSummary({"/points": "sensor_msgs/msg/PointCloud2"})

        with patch("mcap2hdf5.utils.cli_utils.make_reader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_summary.return_value = summary
            mock_reader_cls.return_value = mock_reader

            result = runner.invoke(app, ["inspect", str(mcap)])

        assert "not found" in result.output


class TestCliConfig:
    def testConfigWritesYamlFile(self, tmp_path):
        mcap = tmp_path / "test.mcap"
        mcap.write_bytes(b"")
        summary = _makeChannelSummary(_FULL_TOPICS)

        with patch("mcap2hdf5.utils.cli_utils.make_reader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_summary.return_value = summary
            mock_reader_cls.return_value = mock_reader

            result = runner.invoke(app, ["init", str(mcap)], catch_exceptions=False)

        assert result.exit_code == 0
        config_file = Path("test_config.yaml")
        assert config_file.exists()
        config_file.unlink()

    def testConfigOutputMentionsYamlPath(self, tmp_path):
        mcap = tmp_path / "test.mcap"
        mcap.write_bytes(b"")
        summary = _makeChannelSummary(_FULL_TOPICS)

        with patch("mcap2hdf5.utils.cli_utils.make_reader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_summary.return_value = summary
            mock_reader_cls.return_value = mock_reader

            result = runner.invoke(app, ["init", str(mcap)], catch_exceptions=False)

        assert "_config.yaml" in result.output
        Path("test_config.yaml").unlink(missing_ok=True)

    def testConfigFileNotFound(self):
        result = runner.invoke(app, ["init", "/nonexistent/path/file.mcap"])
        assert result.exit_code != 0

    def testConfigYamlContainsMcapPath(self, tmp_path):
        import yaml

        mcap = tmp_path / "myrecording.mcap"
        mcap.write_bytes(b"")
        summary = _makeChannelSummary(_FULL_TOPICS)

        with patch("mcap2hdf5.utils.cli_utils.make_reader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_summary.return_value = summary
            mock_reader_cls.return_value = mock_reader

            runner.invoke(app, ["init", str(mcap)], catch_exceptions=False)

        config_file = Path("myrecording_config.yaml")
        assert config_file.exists()
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert str(mcap) in data["source"]["mcap"]
        config_file.unlink()


class TestCliNoArgs:
    def testNoArgsShowsHelp(self):
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 2)
        assert "mcap2hdf5" in result.output.lower() or "Usage" in result.output


class TestInspectMcap:
    def testNoSummaryExitsWithError(self, tmp_path):
        import click

        from mcap2hdf5.utils.cli_utils import inspectMcap

        mcap = tmp_path / "nosummary.mcap"
        mcap.write_bytes(b"")

        with patch("mcap2hdf5.utils.cli_utils.make_reader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_summary.return_value = None
            mock_reader_cls.return_value = mock_reader

            with pytest.raises(click.exceptions.Exit):
                inspectMcap(mcap)

    def testFileNotFoundExitsWithError(self, tmp_path):
        import click

        from mcap2hdf5.utils.cli_utils import inspectMcap

        with pytest.raises(click.exceptions.Exit):
            inspectMcap(tmp_path / "missing.mcap")

    def testFallsBackToCountWhenNoStatistics(self, tmp_path):
        from mcap2hdf5.utils.cli_utils import inspectMcap

        mcap = tmp_path / "nostats.mcap"
        mcap.write_bytes(b"")

        channel = SimpleNamespace(id=1, topic="/lidar", schema_id=1)
        schema = SimpleNamespace(id=1, name="sensor_msgs/msg/PointCloud2")
        summary = MagicMock()
        summary.channels = {1: channel}
        summary.schemas = {1: schema}
        summary.statistics = None

        with (
            patch("mcap2hdf5.utils.cli_utils.make_reader") as mock_reader_cls,
            patch(
                "mcap2hdf5.utils.cli_utils.countMessagesByChannel", return_value={1: 42}
            ) as mock_count,
        ):
            mock_reader = MagicMock()
            mock_reader.get_summary.return_value = summary
            mock_reader_cls.return_value = mock_reader

            topicToSchema, topicCounts = inspectMcap(mcap)

        mock_count.assert_called_once_with(mcap)
        assert topicCounts["/lidar"] == 42

    def testReturnsCorrectTopicSchemaMapping(self, tmp_path):
        from mcap2hdf5.utils.cli_utils import inspectMcap

        mcap = tmp_path / "test.mcap"
        mcap.write_bytes(b"")
        summary = _makeChannelSummary({"/lidar": "sensor_msgs/msg/PointCloud2"})

        with patch("mcap2hdf5.utils.cli_utils.make_reader") as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.get_summary.return_value = summary
            mock_reader_cls.return_value = mock_reader

            topicToSchema, topicCounts = inspectMcap(mcap)

        assert topicToSchema["/lidar"] == "sensor_msgs/msg/PointCloud2"
        assert topicCounts["/lidar"] == 10
