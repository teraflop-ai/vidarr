import vidarr

if __name__ == "__main__":
    vidarr.test(
        model_name="timm/efficientvit_b1.r256_in1k",
        checkpoint_path="checkpoints/_b1_r256_in1k_5e-5-re-bicubic/final_model.pt",
        test_dir="/home/henry/Documents/image_datasets/jpeg_experiment/test_data",
        num_classes=2,
        batch_size=512,
        num_threads=12,
        use_compile=False,
        use_scaler=False,
    )
