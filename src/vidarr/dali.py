import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.auto_aug import auto_augment, rand_augment, trivial_augment
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


def random_erase(images):
    images = fn.erase(images)
    return images


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
    images = random_erase(images)
    return images


def apply_training_augmentations(
    images,
    augmentation: str,
    data_config: dict,
):
    images = fn.decoders.image_random_crop(
        images, device="mixed", output_type=types.RGB
    )

    interpolation = {
        "bicubic": types.INTERP_CUBIC,
        "bilinear": types.INTERP_LINEAR,
    }[data_config["interpolation"]]

    image_size = data_config["input_size"][1]
    resize_short = int(image_size / data_config["crop_pct"])

    images = fn.resize(images, resize_shorter=resize_short, interp_type=interpolation)

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
        crop=(image_size, image_size),
        mean=[m * 255 for m in data_config["mean"]],
        std=[s * 255 for s in data_config["std"]],
    )
    return images


def apply_validation_augmentations(
    images,
    data_config: dict,
):
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

    interpolation = {
        "bicubic": types.INTERP_CUBIC,
        "bilinear": types.INTERP_LINEAR,
    }[data_config["interpolation"]]

    image_size = data_config["input_size"][1]
    resize_short = int(image_size / data_config["crop_pct"])

    images = fn.resize(images, resize_shorter=resize_short, interp_type=interpolation)

    images = images.gpu()

    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        crop=(image_size, image_size),
        mean=[m * 255 for m in data_config["mean"]],
        std=[s * 255 for s in data_config["std"]],
    )
    return images


@pipeline_def(enable_conditionals=True)
def dali_training_pipeline(
    images_dir: str,
    augmentation: str,
    data_config: dict,
):
    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=True, pad_last_batch=True, name="Reader"
    )
    images = apply_training_augmentations(
        images=images,
        augmentation=augmentation,
        data_config=data_config,
    )
    return images, labels.gpu()


@pipeline_def
def dali_validation_pipeline(
    images_dir: str,
    data_config: dict,
):
    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=False, pad_last_batch=True, name="Reader"
    )
    images = apply_validation_augmentations(
        images=images,
        data_config=data_config,
    )
    return images, labels.gpu()


def dali_train_loader(
    images_dir: str,
    batch_size: int,
    data_config: dict,
    num_threads: int = 4,
    device_id: int = 0,
    augmentation: str = "default",
):
    pipeline_kwargs = {
        "batch_size": batch_size,
        "num_threads": num_threads,
        "device_id": device_id,
    }
    pipe = dali_training_pipeline(
        images_dir=images_dir,
        augmentation=augmentation,
        data_config=data_config,
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
    data_config: dict,
    num_threads: int = 4,
    device_id: int = 0,
):
    pipeline_kwargs = {
        "batch_size": batch_size,
        "num_threads": num_threads,
        "device_id": device_id,
    }
    pipe = dali_validation_pipeline(
        images_dir=images_dir,
        data_config=data_config,
        **pipeline_kwargs,
    )
    val_loader = DALIClassificationIterator(
        pipe,
        reader_name="Reader",
    )
    return val_loader
