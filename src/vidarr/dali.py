import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.auto_aug import auto_augment

def apply_training_augmentations(images, image_size: int, image_crop: int):
    images = fn.decoders.image_random_crop(
        images, device="mixed", output_type=types.RGB
    )

    images = fn.resize(images, size=image_size)

    images = images.gpu()

    rng = fn.random.coin_flip(probability=0.5)
    images = fn.flip(images, horizontal=rng)

    images = auto_augment.auto_augment_image_net(
        images, shape=[image_size, image_size]
    )

    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        crop=(image_crop, image_crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images


def apply_validation_augmentations(images, image_size: int, image_crop: int):
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

    images = fn.resize(images, size=image_size)

    images = images.gpu()

    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        crop=(image_crop, image_crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images


@pipeline_def(enable_conditionals=True)
def dali_training_pipeline(images_dir: str, image_size: int, image_crop: int):
    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=True, pad_last_batch=True, name="Reader"
    )
    images = apply_training_augmentations(
        images=images, image_size=image_size, image_crop=image_crop
    )
    return images, labels.gpu()


@pipeline_def
def dali_validation_pipeline(images_dir: str, image_size: int, image_crop: int):
    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=False, pad_last_batch=True, name="Reader"
    )
    images = apply_validation_augmentations(
        images=images, image_size=image_size, image_crop=image_crop
    )
    return images, labels.gpu()


def dali_train_loader(
    images_dir: str,
    batch_size: int,
    num_threads: int = 4,
    device_id: int = 0,
    image_size: int = 224,
    image_crop: int = 224,
):
    pipeline_kwargs = {
        "batch_size": batch_size,
        "num_threads": num_threads,
        "device_id": device_id,
    }
    train_loader = DALIClassificationIterator(
        [
            dali_training_pipeline(
                images_dir=images_dir,
                image_size=image_size,
                image_crop=image_crop,
                **pipeline_kwargs,
            )
        ],
        reader_name="Reader",
        fill_last_batch=False,
    )
    return train_loader


def dali_val_loader(
    images_dir: str,
    batch_size: int,
    num_threads: int = 4,
    device_id: int = 0,
    image_size: int = 224,
    image_crop: int = 224,
):
    pipeline_kwargs = {
        "batch_size": batch_size,
        "num_threads": num_threads,
        "device_id": device_id,
    }
    train_loader = DALIClassificationIterator(
        [
            dali_validation_pipeline(
                images_dir=images_dir,
                image_size=image_size,
                image_crop=image_crop,
                **pipeline_kwargs,
            )
        ],
        reader_name="Reader",
        fill_last_batch=False,
    )
    return train_loader
