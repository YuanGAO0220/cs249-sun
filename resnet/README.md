1. Install TensorFlow, Bazel.

2. Download CIFAR-100 dataset.

```shell
curl -o cifar-100-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
```

<b>How to run:</b>

```shell
$ tar -xf cifar-100-binary.tar.gz
$ ls
.:
cifar-100-binary  resnet  WORKSPACE


# Build everything for GPU.
$ bazel build -c opt --config=cuda resnet/...

# Train the model.
$ bazel-bin/resnet/resnet_main --train_data_path=cifar10/train.bin \
                               --log_root=/tmp/resnet_model \
                               --train_dir=/tmp/resnet_model/train \
                               --dataset='cifar100' \
                               --num_gpus=1

# While the model is training, you can also check on its progress using tensorboard:
$ tensorboard --logdir=/tmp/resnet_model

# Evaluate the model.
# Avoid running on the same GPU as the training job at the same time,
# otherwise, you might run out of memory.
$ bazel-bin/resnet/resnet_main --eval_data_path=cifar100/test.bin \
                               --log_root=/tmp/resnet_model \
                               --eval_dir=/tmp/resnet_model/test \
                               --mode=eval \
                               --dataset='cifar100' \
                               --num_gpus=0
```
