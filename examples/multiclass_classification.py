import vidarr

if __name__ == "__main__":
    # "tiny_vit_21m_224" "timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k"
    vidarr.train(
        model_name="timm/efficientvit_m0.r224_in1k",
        train_dir="/home/henry/Documents/image_datasets/jpeg_experiment/train_data",
        val_dir="/home/henry/Documents/image_datasets/jpeg_experiment/val_data",
        num_classes=2,
        num_epochs=150,
        batch_size=1024,
        learning_rate=3e-4,
        scheduler_type="cosine",
        warmup_steps=0.1,
        num_threads=12,
        use_scaler=False,
        use_compile=False,
        metric_type="multiclass",
        criterion_type="crossentropy",
        profiler_dir="./log/efficientvit",
        checkpoint_dir="./checkpoints/_m0_r224_in1k_quick_test",
        use_mixup=True,
        use_re=False,
        augmentation="autoaugment",
    )
