Kaggle :

=> https://www.kaggle.com/c/airbus-ship-detection


python object_detection/model_main.py \
    --pipeline_config_path=/home/ubuntu/kaggle-airbus_ship-faster-RCNN/ssd_mobilenet_v1_ship.config \
    --model_dir=/home/ubuntu/kaggle-airbus_ship-faster-RCNN/model_v2/ \
    --num_train_steps=50000 \
    --num_eval_steps=2000 \
    --alsologtostderr



python object_detection/legacy/train.py --logtostderr --train_dir=/home/ubuntu/kaggle-airbus_ship-faster-RCNN/model/train/ --pipeline_config_path=/home/ubuntu/kaggle-airbus_ship-faster-RCNN/ssd_mobilenet_v1_ship.config 
