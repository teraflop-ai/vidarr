import vidarr

if __name__ == "__main__":
    # "tiny_vit_21m_224" "timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k"
    vidarr.train(
        model_name="timm/efficientvit_m5.r224_in1k",
        train_dir="/home/henry/Documents/image_datasets/jpeg_experiment/train_data",
        val_dir="/home/henry/Documents/image_datasets/jpeg_experiment/val_data",
        num_epochs=20,
        batch_size=1024,
        learning_rate=5.0e-05,
        scheduler_type="cosine",
        warmup_steps=0.10,
        num_threads=12,
        image_size=224,
        image_crop=224,
        use_scaler=False,
        use_compile=True,
        metric_type="binary",
        criterion_type="bcewithlogits",
        profiler_dir="./log/tinyvit",
    )
