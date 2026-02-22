import h5py
import logging
import numpy as np
from pipeline.config import (
    CAMERA,
    CAMERA_D_MATRIX_ATTRIBUTE,
    CAMERA_GROUP,
    CAMERA_HEIGHT_ATTRIBUTE,
    CAMERA_IMAGES_DATASET_PATH,
    CAMERA_K_MATRIX_ATTRIBUTE,
    CAMERA_P_MATRIX_ATTRIBUTE,
    CAMERA_R_MATRIX_ATTRIBUTE,
    CAMERA_WIDTH_ATTRIBUTE,
    CHUNK_ID,
    CHUNK_IDS_DATASET_PATH,
    DATA_COMPRESSION_METHOD,
    DISTORTION_MODEL_ATTRIBUTE,
    INITIAL_LIDAR_CAPACITY,
    LIDAR,
    LIDAR_COUNTS_DATASET_PATH,
    LIDAR_DATA_DATASET_PATH,
    LIDAR_GROUP,
    LIDAR_OFFSETS_DATASET_PATH,
    LIDAR_POINT_OFFSET_ATTRIBUTE,
    NUM_SAMPLES_ATTRIBUTE,
    ROS_MSG,
    SAMPLES_GROUP,
    TIMESTAMP,
    TIMESTAMP_DATASET_PATH,
    TRANSFORMS,
    TRANSFORMS_GROUP,
)
from pipeline.message_converter import MessageConverter

class HDF5Writer:
    def __init__(self, filePath):
        self.filePath = filePath
        self.logger = logging.getLogger(__name__)
        self.h5File = h5py.File(filePath, "w")
        self.initialized = False

    def writeBatch(self, samples):
        if not samples: 
            return
        
        if not self.initialized:
            self.createDatasets(samples[0])
            self.initialized = True

        startIndex = self.h5File.attrs.get(NUM_SAMPLES_ATTRIBUTE,0)
        numNewSamples = len(samples)

        self.resizeDatasets(startIndex + numNewSamples)

        for index, sample in enumerate(samples):
            globalIndex = startIndex + index

            lidarData = MessageConverter.lidarToNumpy(sample[LIDAR][ROS_MSG])
            cameraData = MessageConverter.compressedImageToNumpy(sample[CAMERA][ROS_MSG])

            self.h5File[TIMESTAMP_DATASET_PATH][globalIndex] = sample[TIMESTAMP]
            self.h5File[CHUNK_IDS_DATASET_PATH][globalIndex] = sample[CHUNK_ID]

            pointOffset = self.h5File.attrs.get(LIDAR_POINT_OFFSET_ATTRIBUTE, 0)
            numPoints = lidarData.shape[0]

            if pointOffset + numPoints > self.h5File[LIDAR_DATA_DATASET_PATH].shape[0]:
                self.resizeLidarData(pointOffset+numPoints)
            
            self.h5File[LIDAR_DATA_DATASET_PATH][pointOffset:pointOffset + numPoints] = lidarData
            self.h5File[LIDAR_OFFSETS_DATASET_PATH][globalIndex] = pointOffset
            self.h5File[LIDAR_COUNTS_DATASET_PATH][globalIndex] = numPoints
            self.h5File.attrs[LIDAR_POINT_OFFSET_ATTRIBUTE] = pointOffset + numPoints

            self.h5File[CAMERA_IMAGES_DATASET_PATH][globalIndex] = cameraData

            for tfKey , tfMatrix in sample[TRANSFORMS].items():
                datasetPath = f"{TRANSFORMS_GROUP}/{tfKey}"
                if datasetPath not in self.h5File:
                    self.h5File.create_dataset(
                        datasetPath,
                        shape=(0, 4, 4),
                        maxshape=(None, 4, 4),
                        dtype=np.float32,
                        compression=DATA_COMPRESSION_METHOD,
                    )

                dataset = self.h5File[datasetPath]
                dataset.resize((globalIndex+1,4,4))
                dataset[globalIndex] = tfMatrix

        self.h5File.attrs[NUM_SAMPLES_ATTRIBUTE] = startIndex + numNewSamples


    def createDatasets(self,sampleTemplate):
        self.h5File.create_group(SAMPLES_GROUP)
        self.h5File.create_dataset(
            TIMESTAMP_DATASET_PATH,
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64
        )
        self.h5File.create_dataset(
            CHUNK_IDS_DATASET_PATH,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32
        )
        
        self.h5File.create_group(LIDAR_GROUP)
        self.h5File.create_dataset(
            LIDAR_DATA_DATASET_PATH,
            shape=(INITIAL_LIDAR_CAPACITY, 4),
            maxshape=(None, 4),
            dtype=np.float32,
            compression=DATA_COMPRESSION_METHOD,
            chunks=(10000, 4)
        )
        self.h5File.create_dataset(
            LIDAR_OFFSETS_DATASET_PATH,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64
        )
        self.h5File.create_dataset(
            LIDAR_COUNTS_DATASET_PATH,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32
        )
        
        self.h5File.create_group(CAMERA_GROUP)
        cameraData = MessageConverter.compressedImageToNumpy(sampleTemplate[CAMERA][ROS_MSG])
        height, width, channels = cameraData.shape
        self.h5File.create_dataset(
            CAMERA_IMAGES_DATASET_PATH,
            shape=(0, height, width, channels),
            maxshape=(None, height, width, channels),
            dtype=np.uint8,
            compression=DATA_COMPRESSION_METHOD,
            chunks=(1, height, width, channels)
        )
        
        self.h5File.create_group(TRANSFORMS_GROUP)
        
        self.h5File.attrs[NUM_SAMPLES_ATTRIBUTE] = 0
        self.h5File.attrs[LIDAR_POINT_OFFSET_ATTRIBUTE] = 0

    def resizeDatasets(self, newSize):        
        self.h5File[TIMESTAMP_DATASET_PATH].resize((newSize,))
        self.h5File[CHUNK_IDS_DATASET_PATH].resize((newSize,))
        self.h5File[LIDAR_OFFSETS_DATASET_PATH].resize((newSize,))
        self.h5File[LIDAR_COUNTS_DATASET_PATH].resize((newSize,))
        self.h5File[CAMERA_IMAGES_DATASET_PATH].resize((newSize, 
                                                self.h5File[CAMERA_IMAGES_DATASET_PATH].shape[1], 
                                                self.h5File[CAMERA_IMAGES_DATASET_PATH].shape[2], 
                                                self.h5File[CAMERA_IMAGES_DATASET_PATH].shape[3]))

    def resizeLidarData(self, minSize):
        currentSize = self.h5File[LIDAR_DATA_DATASET_PATH].shape[0]
        newSize = max(minSize, currentSize * 2)
        self.h5File[LIDAR_DATA_DATASET_PATH].resize((newSize, 4))

    def __del__(self):
        if hasattr(self, "h5File") and self.h5File.id.valid:
            self.logger.warning("HDF5Writer was not finalized — closing file in destructor.")
            self.h5File.close()

    def finalize(self, cameraMetadata, staticTransforms):
        numSamples = self.h5File.attrs.get(NUM_SAMPLES_ATTRIBUTE, 0)
        lidarPointOffset = self.h5File.attrs.get(LIDAR_POINT_OFFSET_ATTRIBUTE, 0)
        
        self.h5File[LIDAR_DATA_DATASET_PATH].resize((lidarPointOffset, 4))
        
        if cameraMetadata:
            self.h5File.attrs[CAMERA_K_MATRIX_ATTRIBUTE] = np.array(cameraMetadata.k, dtype=np.float32).reshape(3, 3)
            self.h5File.attrs[CAMERA_D_MATRIX_ATTRIBUTE] = np.array(cameraMetadata.d, dtype=np.float32)
            self.h5File.attrs[CAMERA_R_MATRIX_ATTRIBUTE] = np.array(cameraMetadata.r, dtype=np.float32).reshape(3, 3)
            self.h5File.attrs[CAMERA_P_MATRIX_ATTRIBUTE] = np.array(cameraMetadata.p, dtype=np.float32).reshape(3, 4)
            self.h5File.attrs[DISTORTION_MODEL_ATTRIBUTE] = str(cameraMetadata.distortion_model)
            self.h5File.attrs[CAMERA_WIDTH_ATTRIBUTE] = int(cameraMetadata.width)
            self.h5File.attrs[CAMERA_HEIGHT_ATTRIBUTE] = int(cameraMetadata.height)
        else:
            self.logger.warning("No Camera Metadata found to persist!")

        if staticTransforms:
            staticGroup = self.h5File.create_group("static_transforms")
            for tfStamped in staticTransforms.transforms:
                matrix = MessageConverter.transformToMatrix(tfStamped.transform)
                
                frameId = tfStamped.header.frame_id
                childFrameId = tfStamped.child_frame_id
                key = f"{frameId}_to_{childFrameId}"
                
                staticGroup.create_dataset(key, data=matrix, dtype=np.float32)
        else:
            self.logger.warning("No Static TF found to persist!")
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {numSamples}")
        print(f"  Total lidar points: {lidarPointOffset}")
        avgPoints = lidarPointOffset / numSamples if numSamples > 0 else 0
        print(f"  Average points per sample: {avgPoints:.1f}")

        self.h5File.close()