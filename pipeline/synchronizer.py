from pipeline.config import (
    LIDAR_TOPIC,
    CAMERA_IMAGE_TOPIC,
    TF_TOPIC,
    TF_MATRIX,
    TIMESTAMP,
    LIDAR,
    CAMERA,
    TRANSFORMS,
    ROS_MSG,
    MAX_CHUNK_GAP,
    SENSOR_SYNC_THRESHOLD,
) 
import logging 
from typing import (
    Dict,
    Any,
)
from dataclasses import dataclass
from pipeline.message_converter import MessageConverter
import numpy as np

@dataclass
class SyncGroup:
    timestamp: float
    lidar: Any
    camera: Any
    transforms: Dict[str, Any]

class SensorSynchronizer:
    def __init__(self, syncThreshold, maxGap):
        self.syncThreshold = syncThreshold
        self.maxGap = maxGap
        self.logger = logging.getLogger(__name__)

        self.chunkBuffer = {
            LIDAR_TOPIC: [],
            CAMERA_IMAGE_TOPIC: [],
        }

        self.lastTimestamps = {
            LIDAR_TOPIC: None,
            CAMERA_IMAGE_TOPIC: None
        }

        self.tfCache = {}
        self.staticTransforms = None

    def processMessage(self,topic,rosMsg, timestamp):
        
        chunkEntry = {
            TIMESTAMP: timestamp,
            ROS_MSG: rosMsg,
        }
        
        if topic == TF_TOPIC: 
            self.updateTFCache(chunkEntry)
            return

        if topic in [LIDAR_TOPIC, CAMERA_IMAGE_TOPIC]:
            if self.checkFlushConstraint(topic, timestamp):
                yield from self.flushSamples()
            
            self.chunkBuffer[topic].append(chunkEntry)
            self.lastTimestamps[topic] = timestamp

    def updateTFCache(self, tfEntry):
        for tfStamped in tfEntry[ROS_MSG].transforms:
            timestamp = tfEntry[TIMESTAMP]
            frameId = tfStamped.header.frame_id
            childFrameId = tfStamped.child_frame_id
            key = f"{frameId}_to_{childFrameId}"
            
            if key not in self.tfCache:
                self.tfCache[key] = []
            
            matrix = MessageConverter.transformToMatrix(tfStamped.transform)
            self.tfCache[key].append({
                TIMESTAMP: timestamp,
                TF_MATRIX: matrix
            })

            if len(self.tfCache[key]) > 100:
                self.tfCache[key].pop(0)


    def checkFlushConstraint(self, sensorTopicName, currentTimestamp):
        lastTimestamp = self.lastTimestamps.get(sensorTopicName)
        if lastTimestamp is None:
            return False
        return (currentTimestamp - lastTimestamp) > MAX_CHUNK_GAP

    def flushSamples(self):
        lidarFrames = self.chunkBuffer[LIDAR_TOPIC]
        cameraFrames = self.chunkBuffer[CAMERA_IMAGE_TOPIC]
        samples = []

        for lidarEntry in lidarFrames:
            """ Use LIDAR timestamp as the reference """
            lidarTimestamp = lidarEntry[TIMESTAMP]
            closestCamera = self.findClosestFrame(lidarTimestamp, cameraFrames)

            if closestCamera is None:
                continue
            
            timeDiff = abs(lidarTimestamp - closestCamera[TIMESTAMP])

            if timeDiff > SENSOR_SYNC_THRESHOLD:
                continue
            
            transforms = self.interpolateTransforms(lidarTimestamp)
            
            sample = {
                LIDAR: lidarEntry,
                CAMERA: closestCamera,
                TRANSFORMS: transforms,
                TIMESTAMP: lidarTimestamp,
            }
            samples.append(sample)
        
        self.chunkBuffer = {
            LIDAR_TOPIC: [],
            CAMERA_IMAGE_TOPIC: [],
            TF_TOPIC: [],
        }
        self.lastTimestamps = {key: None for key in self.lastTimestamps}
        
        for sample in samples:
            yield sample

    def findClosestFrame(self, targetTimestamp, frames):
        if not frames:
            return None
        
        closestFrame = None
        minDiff = float('inf')
        
        for frame in frames:
            diff = abs(frame[TIMESTAMP] - targetTimestamp)
            if diff < minDiff:
                minDiff = diff
                closestFrame = frame
        
        return closestFrame

    def interpolateTransforms(self, targetTimestamp):
        transforms = {}
        
        for key, tfList in self.tfCache.items():
            if not tfList:
                continue
            
            beforeIdx = None
            afterIdx = None
            
            for index, tf in enumerate(tfList):
                if tf[TIMESTAMP] <= targetTimestamp:
                    beforeIdx = index
                if tf[TIMESTAMP] >= targetTimestamp and afterIdx is None:
                    afterIdx = index
                    break
            
            if beforeIdx is not None and afterIdx is not None:
                if beforeIdx == afterIdx:
                    transforms[key] = tfList[beforeIdx][TF_MATRIX]
                else:
                    before = tfList[beforeIdx]
                    after = tfList[afterIdx]
                    
                    alpha = (targetTimestamp - before[TIMESTAMP]) / (after[TIMESTAMP] - before[TIMESTAMP])
                    alpha = np.clip(alpha, 0.0, 1.0)
                    
                    transforms[key] = self.interpolateMatrix(before[TF_MATRIX], after[TF_MATRIX], alpha)
            
            elif beforeIdx is not None:
                transforms[key] = tfList[beforeIdx][TF_MATRIX]
            elif afterIdx is not None:
                transforms[key] = tfList[afterIdx][TF_MATRIX]
        
        return transforms