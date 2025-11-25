import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def apply_training_augmentations(images, image_size):
    images = fn.decoders.image_random_crop(
        images, device="mixed", output_type=types.RGB
    )

    images = fn.resize(images, size=image_size)

    images = images.gpu()

    rng = fn.random.coin_flip(probability=0.5)
    images = fn.flip(images, horizontal=rng, vertical=rng)

    images = fn.crop_mirror_normalize(
        images,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip(),
    )
    return images


@pipeline_def(num_threads=4, device_id=0)
def dali_training_pipeline(images_dir: str, image_size: int = 224):
    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=True, pad_last_batch=True, name="Reader"
    )
    images = apply_training_augmentations(images=images, image_size=image_size)
    return images, labels.gpu()


def dali_train_loader(images_dir: str, batch_size: int = 128):
    pipeline_kwargs = {"batch_size": batch_size}
    train_loader = DALIGenericIterator(
        [
            dali_training_pipeline(
                images_dir=images_dir,
                **pipeline_kwargs,
            )
        ],
        ["data", "label"],
        reader_name="Reader",
    )
    return train_loader
