MODEL=ssd_mobilenet_v2_320x320_coco17_tpu-8
EXPERIMENT=run1
PIPELINE_CONFIG_PATH=/home/ubuntu/balloon_project/models/${MODEL}/${EXPERIMENT}/pipeline.config
MODEL_DIR=/home/ubuntu/balloon_project/models/${MODEL}/${EXPERIMENT}/
TRAIN_CHECKPOINT_DIR=/home/ubuntu/balloon_project/models/${MODEL}/${EXPERIMENT}/
EXPORT_DIR=/home/ubuntu/balloon_project/exported_models/${MODEL}/tflite/

python export_tflite_graph_tf2.py \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_dir ${TRAIN_CHECKPOINT_DIR} \
    --output_directory ${EXPORT_DIR}
    
tflite_convert --saved_model_dir=${EXPORT_DIR}/saved_model --output_file=${EXPORT_DIR}/model.tflite