# vidarr

## Installation
```
pip install vidarr
```
## Usage

### Train your classifier
```python
import vidarr

if __name__ == "__main__":
    vidarr.train(
        model_name="timm/efficientvit_b1.r288_in1k",
        train_dir="/image_datasets/jpeg_experiment/train_data",
        val_dir="/image_datasets/jpeg_experiment/val_data",
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
        use_mixup=True,
        augmentation="trivialaugment",
    )
```
### Analyze the profiling trace
```python
import vidarr

if __name__ == "__main__":
    vidarr.analyze_run(
        breakdown="temporal",
        trace_dir="/log/tinyvit"
    )
```
### Test the trained model
```python
import vidarr

if __name__ == "__main__":
    vidarr.test(
        model_name="timm/efficientvit_b1.r288_in1k",
        checkpoint_path="/checkpoints/final_model.pt",
        test_dir="/image_datasets/jpeg_experiment/test_data",
        num_classes=2,
        batch_size=512,
        image_size=384,
        crop_size=384,
        num_threads=12,
        use_compile=True,
    )
```

## Contributing
```python
uv pip install -e .
```
```
ruff check --select I --fix .
ruff format .
```