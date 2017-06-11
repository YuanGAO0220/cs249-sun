bazel-bin/resnet/resnet_main --eval_data_path=cifar-100-binary/test.bin \
                               --log_root=/tmp/resnet_model \
                               --eval_dir=/tmp/resnet_model/test \
                               --mode=eval \
                               --dataset='cifar100' \
                               --num_gpus=0