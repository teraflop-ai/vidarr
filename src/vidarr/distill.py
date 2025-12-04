import albumentations as A
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model="timm/efficientvit_m0.r224_in1k",
        method="distillation",
        epochs=100,
        batch_size=128,
    )
