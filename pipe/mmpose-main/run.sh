python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
    configs/face_2d_keypoint/rtmpose/face6/rtmpose-m_8xb256-120e_face6-256x256.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth \
    --input demo_face.mp4 \
    --output-root vis_results --radius 1