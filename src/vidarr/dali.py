import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.auto_aug import auto_augment, rand_augment, trivial_augment
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


def random_color_jitter(images, p=0.05):
    brightness = fn.random.uniform(range=(0.8, 1.2))
    contrast = fn.random.uniform(range=(0.8, 1.2))
    saturation = fn.random.uniform(range=(0.8, 1.3))
    hue = fn.random.uniform(range=(-0.05, 0.05))

    color_jitter = fn.color_twist(
        images,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    do = fn.random.coin_flip(probability=p)
    return do * color_jitter + (1 - do) * images


def random_gaussian_blur(images, p=0.05, sigma_range=(0.1, 2.0)):
    sigma = fn.random.uniform(range=sigma_range)
    gaussian_blur = fn.gaussian_blur(images, sigma=sigma)
    do = fn.random.coin_flip(probability=p)
    images = do * gaussian_blur + (1.0 - do) * images
    return images


def default_augmentations(images):
    images = random_color_jitter(images)
    images = random_gaussian_blur(images)
    return images


def apply_training_augmentations(
    images, image_size: int, image_crop: int, augmentation: str = "trivialaugment"
):
    images = fn.decoders.image_random_crop(
        images, device="mixed", output_type=types.RGB
    )

    images = fn.resize(images, size=image_size)

    images = images.gpu()

    rng = fn.random.coin_flip(probability=0.5)
    images = fn.flip(images, horizontal=rng)

    if augmentation == "autoaugment":
        images = auto_augment.auto_augment_image_net(
            images, shape=[image_size, image_size]
        )
    elif augmentation == "trivialaugment":
        images = trivial_augment.trivial_augment_wide(
            images, shape=[image_size, image_size]
        )
    elif augmentation == "randaugment":
        images = rand_augment.rand_augment(images, shape=[image_size, image_size])
    elif augmentation == "default":
        images = default_augmentations(images)
    else:
        raise NotImplementedError()

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
    pipe = dali_training_pipeline(
        images_dir=images_dir,
        image_size=image_size,
        image_crop=image_crop,
        **pipeline_kwargs,
    )
    train_loader = DALIClassificationIterator(
        pipe,
        reader_name="Reader",
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
    pipe = dali_validation_pipeline(
        images_dir=images_dir,
        image_size=image_size,
        image_crop=image_crop,
        **pipeline_kwargs,
    )
    val_loader = DALIClassificationIterator(
        pipe,
        reader_name="Reader",
    )
    return val_loader
