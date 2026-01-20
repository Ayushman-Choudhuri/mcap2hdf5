import cv2
import h5py
from mcap_ros2.reader import read_ros2_messages
import numpy as np
import os

MCAP_FILE_PATH = "data/raw/kitti.mcap"
CHUNKS_FILE_PATH = "data/processed/chunks.hdf5"

TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
LIDAR_TOPIC = "/sensor/lidar/front/points"
CAMERA_INTRINSIC_PARAMETERS_TOPIC = "/sensor/camera/left/camera_info"
CAMERA_IMAGE_TOPIC = "/sensor/camera/left/image_raw/compressed"
TIMESTAMP = "timestamp"
ROS_MSG = "rosMsg"

class MCAPHandler:
    def __init__(self, mcapFilePath , maxChunkGap=0.15):
        self.mcapFilePath = mcapFilePath
        self.maxChunkGap = maxChunkGap
        self.chunkBuffer = {
            LIDAR_TOPIC:[],
            CAMERA_IMAGE_TOPIC: [],
            CAMERA_INTRINSIC_PARAMETERS_TOPIC:[],
            TF_TOPIC:[],
            TF_STATIC_TOPIC:[],
        }
        self.lastTimestamps = {
            LIDAR_TOPIC: None,
            CAMERA_IMAGE_TOPIC:None
        }

    def __iter__(self):
        return self.generateChunk()

    def generateChunk(self):
        with open(self.mcapFilePath, "rb") as f:
            for msg in read_ros2_messages(f):
                topic = msg.channel.topic
                rosMsg = msg.ros_msg
                timestamp = None
                if hasattr(rosMsg, "header") and hasattr(rosMsg.header, "stamp"):
                    timestamp = rosMsg.header.stamp.sec + (rosMsg.header.stamp.nanosec *1e-9)

                chunkEntry = {
                    TIMESTAMP : timestamp,
                    ROS_MSG: rosMsg,
                }
                if topic == TF_TOPIC:
                    self.chunkBuffer[TF_TOPIC].append(chunkEntry)
                elif topic == TF_STATIC_TOPIC:
                    self.chunkBuffer[TF_STATIC_TOPIC].append(chunkEntry)
                elif topic == CAMERA_INTRINSIC_PARAMETERS_TOPIC:
                    self.chunkBuffer[CAMERA_INTRINSIC_PARAMETERS_TOPIC].append(chunkEntry)
                elif topic == LIDAR_TOPIC:
                    if self.checkFlushConstraint(LIDAR_TOPIC, timestamp):
                        yield self.flushChunk()
                    self.chunkBuffer[LIDAR_TOPIC].append(chunkEntry)
                    self.lastTimestamps[LIDAR_TOPIC] = timestamp
                elif topic == CAMERA_IMAGE_TOPIC:
                    if self.checkFlushConstraint(CAMERA_IMAGE_TOPIC, timestamp):
                        yield self.flushChunk()
                    self.chunkBuffer[CAMERA_IMAGE_TOPIC].append(chunkEntry)
                    self.lastTimestamps[CAMERA_IMAGE_TOPIC] = timestamp
                else:
                    continue

    def flushChunk(self ):
        chunk = {topic : info[:] for topic, info in self.chunkBuffer.items()}
        self.chunkBuffer = {
            LIDAR_TOPIC:[],
            CAMERA_IMAGE_TOPIC: [],
            CAMERA_INTRINSIC_PARAMETERS_TOPIC:[],
            TF_TOPIC:[],
            TF_STATIC_TOPIC:[],
        }
        self.lastTimestamps = {key : None for key in self.lastTimestamps}
        return chunk

    def checkFlushConstraint(self, sensorTopicName, currentTimestamp):
        lastTimestamp = self.lastTimestamps.get(sensorTopicName)
        if lastTimestamp is None:
            return False
        gap = currentTimestamp - lastTimestamp
        if gap > self.maxChunkGap:
            return True
        return False

    @staticmethod
    def lidarToNumpy(lidarMsg):
        dtypeMap = {
            1: np.int8,
            2: np.uint8,
            3: np.int16,
            4: np.uint16,
            5: np.int32,
            6: np.uint32,
            7: np.float32,
            8: np.float64,
        }

        fields = []
        for field in lidarMsg.fields:
            if field.name in ("x", "y", "z", "intensity"):
                fields.append(
                    (field.name, 
                     dtypeMap[field.datatype])
                )

        if len(fields) < 4:
            raise ValueError("PointCloud2 does not contain x,y,z,intensity")

        cloud = np.frombuffer(lidarMsg.data, dtype=np.dtype(fields))

        return np.stack(
            (cloud["x"], 
             cloud["y"], 
             cloud["z"], 
             cloud["intensity"]
            ),
            axis=-1
        ).astype(np.float32)
    
    @staticmethod
    def compressedImageToNumpy(imageMsg):
        imageArray = np.frombuffer(imageMsg.data, np.uint8)
        cv2Image = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)
        return cv2Image
    
    def exportChunkToHDF5(self, chunk, chunkIndex, h5File): 
        chunkGroup = h5File.create_group(f"chunk_{chunkIndex:05d}")
        
        for topic, messages in chunk.items():
            if not messages:
                continue

            if topic not in [LIDAR_TOPIC, 
                             CAMERA_IMAGE_TOPIC, 
                             CAMERA_INTRINSIC_PARAMETERS_TOPIC]:
                continue

            cleanTopic = topic.strip("/")
            topicGroup = chunkGroup.create_group(cleanTopic)

            for index, entry in enumerate(messages):
                rosMsg = entry[ROS_MSG]
                timestamp = entry[TIMESTAMP]

                frameGroup = topicGroup.create_group(f"frame_{index:03d}")
                frameGroup.attrs["timestamp"] = timestamp

                if topic == LIDAR_TOPIC:
                    data = self.lidarToNumpy(rosMsg)
                    frameGroup.create_dataset("data", data=data, compression="lzf")

                elif topic == CAMERA_IMAGE_TOPIC:
                    data = self.compressedImageToNumpy(rosMsg)
                    frameGroup.create_dataset("data", data=data, compression="lzf")

                elif topic == CAMERA_INTRINSIC_PARAMETERS_TOPIC:
                    k_matrix = np.array(rosMsg.k, dtype=np.float32).reshape(3, 3)
                    frameGroup.create_dataset("k", data=k_matrix)

if __name__ == "__main__":
    parser = MCAPHandler(MCAP_FILE_PATH)
    os.makedirs(os.path.dirname(CHUNKS_FILE_PATH), exist_ok=True)

    with h5py.File(CHUNKS_FILE_PATH, "w") as h5File:
        print(f"Starting Conversion: {MCAP_FILE_PATH} -> {CHUNKS_FILE_PATH}")
        for index, chunk in enumerate(parser):
            try:
                parser.exportChunkToHDF5(chunk, index, h5File)
                
                if index % 10 == 0:
                    print(f"Processed chunk {index}...")
            
            except Exception as e:
                print(f"Error processing chunk {index}: {e}")
                continue

    print("Conversion complete!")

