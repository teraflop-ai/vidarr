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
        model_name="timm/efficientvit_m5.r224_in1k",
        train_dir="/image_datasets/jpeg_experiment/train_data",
        val_dir="/image_datasets/jpeg_experiment/val_data",
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

## Contributing
```python
uv pip install -e .
```
```
ruff check --select I --fix .
ruff format .
```