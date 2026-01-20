import cv2
import h5py
from mcap_ros2.reader import read_ros2_messages
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

MCAP_FILE_PATH = "data/raw/kitti.mcap"
CHUNKS_FILE_PATH = "data/processed/chunks.hdf5"

MAX_CHUNK_GAP = 0.15

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

        self.staticCache = {
            TF_STATIC_TOPIC: None,
            CAMERA_INTRINSIC_PARAMETERS_TOPIC:None,
        }
        self.chunkBuffer = {
            LIDAR_TOPIC:[],
            CAMERA_IMAGE_TOPIC: [],
            TF_TOPIC:[],
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
                timestamp = msg.log_time.timestamp() 
                
                if hasattr(rosMsg, "header") and hasattr(rosMsg.header, "stamp"):
                    timestamp = rosMsg.header.stamp.sec + (rosMsg.header.stamp.nanosec * 1e-9)
                
                elif topic in [TF_TOPIC, TF_STATIC_TOPIC] and len(rosMsg.transforms) > 0:
                    t0 = rosMsg.transforms[0]
                    timestamp = t0.header.stamp.sec + (t0.header.stamp.nanosec * 1e-9)

                chunkEntry = {
                    TIMESTAMP: timestamp,
                    ROS_MSG: rosMsg,
                }

                if topic in self.staticCache and self.staticCache[topic] is None:
                    self.staticCache[topic] = chunkEntry

                if topic == TF_TOPIC:
                    self.chunkBuffer[TF_TOPIC].append(chunkEntry)
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
            TF_TOPIC:[],
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
    
    def transformToMatrix(self,transform):
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
    
    def exportChunkToHDF5(self, chunk, chunkIndex, h5File):
        chunkName = f"chunk_{chunkIndex:05d}"
        chunkGroup = h5File.create_group(chunkName)

        for topic in [LIDAR_TOPIC, CAMERA_IMAGE_TOPIC, TF_TOPIC]:
            messages = chunk.get(topic)
            if not messages:
                continue

            topicGroup = chunkGroup.create_group(topic.strip("/"))
            for index, entry in enumerate(messages):
                frameGroup = topicGroup.create_group(f"frame_{index:03d}")
                frameGroup.attrs["timestamp"] = float(entry[TIMESTAMP])

                if topic == LIDAR_TOPIC:
                    frameGroup.create_dataset("data", data=self.lidarToNumpy(entry[ROS_MSG]), compression="lzf")
                elif topic == CAMERA_IMAGE_TOPIC:
                    frameGroup.create_dataset("data", data=self.compressedImageToNumpy(entry[ROS_MSG]), compression="lzf")
                elif topic == TF_TOPIC:
                    for tfIndex, tfStamp in enumerate(entry[ROS_MSG].transforms):
                        matrix = self.transformToMatrix(tfStamp.transform)
                        dataset = frameGroup.create_dataset(f"pose_{tfIndex}", data=matrix)
                        dataset.attrs["frameId"] = str(tfStamp.header.frame_id)
                        dataset.attrs["child_frame_id"] = str(tfStamp.child_frame_id)


if __name__ == "__main__":
    parser = MCAPHandler(MCAP_FILE_PATH, MAX_CHUNK_GAP)
    
    with h5py.File(CHUNKS_FILE_PATH, "w") as h5File:
        print(f"Starting Conversion...")
        for index, chunk in enumerate(parser):
            parser.exportChunkToHDF5(chunk, index, h5File)
            print(f"Processed chunk {index}...")

        print("\nWriting global metadata to root...")

        cameraMetadata = parser.staticCache.get(CAMERA_INTRINSIC_PARAMETERS_TOPIC)
        if cameraMetadata:
            msg = cameraMetadata[ROS_MSG]
            h5File.attrs["camera_k"] = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            h5File.attrs["camera_d"] = np.array(msg.d, dtype=np.float32)
            h5File.attrs["camera_r"] = np.array(msg.r, dtype=np.float32).reshape(3, 3)
            h5File.attrs["camera_p"] = np.array(msg.p, dtype=np.float32).reshape(3, 4)
            h5File.attrs["distortion_model"] = str(msg.distortion_model)
            h5File.attrs["width"] = int(msg.width)
            h5File.attrs["height"] = int(msg.height)
        else:
            print("Warning: No Camera Metadata found in the MCAP file!")

        staticTransforms = parser.staticCache.get(TF_STATIC_TOPIC)
        if staticTransforms:
            for tf_stamped in staticTransforms[ROS_MSG].transforms:
                matrix = parser.transformToMatrix(tf_stamped.transform)
                key = f"static_tf_{tf_stamped.header.frame_id}_to_{tf_stamped.child_frame_id}"
                h5File.attrs[key] = matrix
        else:
            print("Warning: No Static TF found in the MCAP file!")

    print("\nConversion complete!")