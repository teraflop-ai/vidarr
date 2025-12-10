import vidarr

if __name__ == "__main__":
    vidarr.test(
        model_name="timm/efficientvit_b1.r288_in1k",
        checkpoint_path="/home/henry/vidarr/checkpoints/final_model.pt",
        test_dir="/home/henry/Documents/image_datasets/jpeg_experiment/test_data",
        num_classes=2,
        batch_size=512,
        image_size=384,
        crop_size=384,
        num_threads=12,
        use_compile=True,
    )
