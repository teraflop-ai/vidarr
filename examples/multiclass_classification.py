import vidarr

if __name__ == "__main__":
    # "tiny_vit_21m_224" "timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k"
    vidarr.train(
        model_name="timm/efficientvit_b1.r288_in1k",
        train_dir="/home/henry/Documents/image_datasets/jpeg_experiment/train_data",
        val_dir="/home/henry/Documents/image_datasets/jpeg_experiment/val_data",
        num_classes=2,
        num_epochs=3,
        batch_size=256,
        learning_rate=5.0e-05,
        scheduler_type="cosine",
        warmup_steps=0.10,
        num_threads=12,
        image_size=288,
        image_crop=288,
        use_scaler=False,
        use_compile=True,
        metric_type="multiclass",
        criterion_type="crossentropy",
        profiler_dir="./log/efficientvit",
        checkpoint_dir="./checkpoint",
        use_mixup=True,
        augmentation="trivialaugment",
    )
