import logging

from mcap_ros2.reader import read_ros2_messages

from mcap2hdf5.config import (
    CAMERA_INTRINSIC_PARAMETERS_TOPIC,
    TF_STATIC_TOPIC,
)
from mcap2hdf5.dataclasses import StreamMessage

class MCAPSource:
    def __init__(self,dataSourcePath):
        self.dataSourcePath = dataSourcePath
        self.cameraMetadata = None
        self.staticTransforms = None
        self.logger = logging.getLogger(__name__)

    def streamMessages(self):
        try:
            with open(self.dataSourcePath, "rb") as f:
                for msg in read_ros2_messages(f):
                    topic = msg.channel.topic
                    rosMsg = msg.ros_msg

                    timestamp = self.extractTimestamp(
                        rosMsg,
                        msg.log_time.timestamp()
                    )

                    if topic == CAMERA_INTRINSIC_PARAMETERS_TOPIC and self.cameraMetadata is None:
                        self.cameraMetadata= rosMsg
                        self.logger.info(f"Captured camera metadata from {topic}")

                    if topic == TF_STATIC_TOPIC and self.staticTransforms is None:
                        self.staticTransforms = rosMsg
                        self.logger.info(f"Captured static transforms from {topic}")

                    yield StreamMessage(
                        topic=topic,
                        msg=rosMsg,
                        timestamp=timestamp,
                    )

        except FileNotFoundError:
            self.logger.error(f"MCAP file not found: {self.dataSourcePath}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading MCAP stream: {e}")
            raise

    def extractTimestamp(self, rosMsg, mcapLogTime):
        if hasattr(rosMsg, "header") and hasattr(rosMsg.header, "stamp"):
            return rosMsg.header.stamp.sec + (rosMsg.header.stamp.nanosec * 1e-9)

        if hasattr(rosMsg, "transforms") and len(rosMsg.transforms) > 0:
            t0 = rosMsg.transforms[0]
            return t0.header.stamp.sec + (t0.header.stamp.nanosec * 1e-9)

        return mcapLogTime

    def getCameraMetadata(self):
        return self.cameraMetadata

    def getStaticTransforms(self):
        return self.staticTransforms
