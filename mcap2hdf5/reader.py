from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from mcap2hdf5.utils.dataclasses import StreamMessage
from mcap2hdf5.utils.logger import logger


class MCAPSource:
    def __init__(
        self,
        dataSourcePath,
        topics: list[str] | None = None,
        cameraInfoTopic: str | None = None,
        tfStaticTopic: str | None = None,
    ):
        self.dataSourcePath = dataSourcePath
        self.topics = topics
        self.cameraInfoTopic = cameraInfoTopic
        self.tfStaticTopic = tfStaticTopic
        self.cameraMetadata = None
        self.staticTransforms = None

    def streamMessages(self):
        try:
            with open(self.dataSourcePath, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                for _, channel, message, rosMsg in reader.iter_decoded_messages(
                    topics=self.topics
                ):
                    topic = channel.topic
                    timestamp = self.extractTimestamp(rosMsg, message.log_time / 1e9)

                    if (
                        self.cameraInfoTopic
                        and topic == self.cameraInfoTopic
                        and self.cameraMetadata is None
                    ):
                        self.cameraMetadata = rosMsg
                        logger.info(f"Captured camera metadata from {topic}")

                    if (
                        self.tfStaticTopic
                        and topic == self.tfStaticTopic
                        and self.staticTransforms is None
                    ):
                        self.staticTransforms = rosMsg
                        logger.info(f"Captured static transforms from {topic}")

                    yield StreamMessage(
                        topic=topic,
                        msg=rosMsg,
                        timestamp=timestamp,
                    )

        except FileNotFoundError:
            logger.error(f"MCAP file not found: {self.dataSourcePath}")
            raise
        except Exception as e:
            logger.error(f"Error reading MCAP stream: {e}")
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
