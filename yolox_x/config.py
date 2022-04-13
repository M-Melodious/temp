from types import SimpleNamespace as namespace


inference = namespace(
    model="yolox_x/models/INT8/yolox_x.xml", # Path to model xml file
    batch_size=4,                      # Batch size, equal to num of vid sources, default: 4
    num_requests=2,                    # Num requests for async mode
    mode="async",                      # Inference Mode, default: async
    device="CPU"                       # Device, default: CPU
)

vid_sources = namespace(
    source_dir="yolox_x/videos/"             # Path to input videos
)

images = namespace(
    source_dir="images/"             # Path to input videos
)

post_process = namespace(
    nms_thr=0.45,                      # NMS threshold, default: 0.45
    score_thr=0.3                      # Score threshold, default: 0.3
)

visualization = namespace(
    output_dir="./data",        # Path to output dir, default: ./demo_output
    max_window_size=(1920, 1080),      # Max window size to display, default: (1920, 1080)
    target_size=(1280, 960)            # Target size for video writer, default: (1280, 960)
)
