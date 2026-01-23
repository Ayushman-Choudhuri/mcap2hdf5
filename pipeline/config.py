MCAP_FILE_PATH = "data/raw/kitti.mcap"
CHUNKS_FILE_PATH = "data/processed/chunks.hdf5"


MAX_CHUNK_GAP = 0.15
SENSOR_SYNC_THRESHOLD = 0.05
HDF5_WRITE_BATCH_SIZE = 100

CAMERA_INTRINSIC_PARAMETERS_TOPIC = "/sensor/camera/left/camera_info"
LIDAR_TOPIC = "/sensor/lidar/front/points"
CAMERA_IMAGE_TOPIC = "/sensor/camera/left/image_raw/compressed"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"


TIMESTAMP = "timestamp"
ROS_MSG = "rosMsg"
TF_MATRIX = "tfMatrix"
LIDAR = "lidar"
CAMERA = "camera"
TRANSFORMS = "transforms"
CHUNK_ID = "chunkId"

SAMPLES_GROUP = "samples"
LIDAR_GROUP = "lidar"
CAMERA_GROUP = "camera"
TRANSFORMS_GROUP = "transforms"

TIMESTAMP_DATASET_PATH = SAMPLES_GROUP + "/timestamps"
CHUNK_IDS_DATASET_PATH = SAMPLES_GROUP + "/chunk_ids"
LIDAR_DATA_DATASET_PATH = LIDAR_GROUP + "/data"
LIDAR_OFFSETS_DATASET_PATH = LIDAR_GROUP + "/offsets"
LIDAR_COUNTS_DATASET_PATH = LIDAR_GROUP + "/counts"
CAMERA_IMAGES_DATASET_PATH = CAMERA_GROUP + "/images"

INITIAL_LIDAR_CAPACITY = 1000000
DATA_COMPRESSION_METHOD = "lzf"

NUM_SAMPLES_ATTRIBUTE = "num_samples"
LIDAR_POINT_OFFSET_ATTRIBUTE = "lidar_point_offset"

CAMERA_K_MATRIX_ATTRIBUTE = "camera_k"
CAMERA_D_MATRIX_ATTRIBUTE = "camera_d"
CAMERA_R_MATRIX_ATTRIBUTE = "camera_r"
CAMERA_P_MATRIX_ATTRIBUTE = "camera_p"
DISTORTION_MODEL_ATTRIBUTE = "distortion_model"
CAMERA_WIDTH_ATTRIBUTE = "camera_width"
CAMERA_HEIGHT_ATTRIBUTE = "camera_height"