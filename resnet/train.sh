bazel-bin/resnet/resnet_main --train_data_path=cifar-100-binary/train.bin \
                               --log_root=/tmp/resnet_model \
                               --train_dir=/tmp/resnet_model/train \
                               --dataset='cifar100' \
                               --num_gpus=1