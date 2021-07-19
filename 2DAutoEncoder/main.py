from os.path import join
from train import train
from test import test


def main(cfg):
    if "train" in cfg.mode:
        train(cfg)

    if "test" in cfg.mode:
        test(cfg)


class config:
    mode = ["train", "test", "save_input"]

    # save path
    save_root_path = "result/test4"
    checkpoints_path = join(save_root_path, 'checkpoints')
    sample_path = join(save_root_path, "sample")
    activation_path = join(save_root_path, "activation")
    log_path = join(save_root_path, "logs")
    input_sample_path = join(save_root_path, "input_sample")
    xlsx_path = join(save_root_path, "history.xlsx")

    # dataset
    train_x_path = '../dataset2D/ReconData5set/256x256pix/train/image'
    train_y_path = '../dataset2D/ReconData5set/256x256pix/train/label'
    test_x_path = '../dataset2D/ReconData5set/256x256pix/test/image'
    test_y_path = '../dataset2D/ReconData5set/256x256pix/test/label'

    # for train parameter
    im_dim = (256, 256, 1)  # 入力データサイズ
    epochs = 500
    batch_size = 80
    Adam_lr = 1e-4
    Adam_beta = 0.9
    # clip_dim = (32, 32, 1)  # 入力データからクリップするサイズ = モデルの入力サイズ
    clip_dim = None
    clip_num = 10
    train_weight_path = None
    validation_split = 0.1

    # use only for testing
    test_weight_path = join(save_root_path, "weight.h5")


if __name__ == "__main__":
    cfg = config()
    main(cfg)
