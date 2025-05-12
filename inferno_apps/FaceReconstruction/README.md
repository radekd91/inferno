# In-the-wild 3D Face Reconstruction

This project contains EMICA - a combination of [DECA](https://deca.is.tue.mpg.de/), [EMOCA](https://emoca.is.tue.mpg.de/), [SPECTRE](https://filby89.github.io/spectre/) and [MICA](https://zielon.github.io/mica/) which produces excellent results.

## Download models  

Run `bash download_assets.sh` to download the models and sample data.

## Run demos 

### Image reconstruction demo 

To run Face Reconstruction on a folder with images run: 

```bash 
python demo_face_rec_on_images.py
```

See the script for additional details


### Video reconstruction demo 

To run Face Reconstruction on a folder with images run: 

```bash 
python demo_face_rec_on_video.py
```

See the script for additional details

## Configuration Guide

The Face Reconstruction system offers various configuration options to tailor the reconstruction process to your needs. Below is a comprehensive explanation of the available parameters.

### Quality Settings

#### Detail Mode
- `mode = 'coarse'` or `mode = 'detail'` (in code)
- Controls the level of detail in the 3D face reconstruction
- **Coarse**: Faster processing with less detail
- **Detail**: Higher quality reconstruction with more facial details but slower processing

#### Model Selection
- `--model_name` parameter (default: 'EMICA-CVT_flame2020_notexture')
- Different models offer varying reconstruction qualities
- Options include EMOCA, DECA, EMICA variants

### GPU and Performance Configuration

#### Batch Size
- `batch_size=4` in the TestFaceVideoDM initialization
- Controls how many frames are processed together
- Increasing may improve speed but requires more GPU memory

#### Number of Workers
- `num_workers=4` in the TestFaceVideoDM initialization
- Controls parallel data loading threads
- Increasing may improve data loading speed but uses more CPU resources

#### Device Selection
- The model is loaded with `model.cuda()` which uses the default CUDA device
- To specify a particular GPU, modify this line with a specific device ID (e.g., `model.cuda(device=0)`)

### Output Configuration

#### Output Types
- `--save_images` (default: True): Save output visualization images
- `--save_codes` (default: False): Save FLAME shape, expression, jaw pose values
- `--save_mesh` (default: False): Save 3D meshes of the reconstructed faces

#### Image Type
- `--image_type` (default: 'geometry')
- Controls what type of reconstruction image to create

#### Video Creation Options
- `--include_rec` (default: True): Include non-transparent reconstruction in the video
- `--include_transparent` (default: True): Add transparent mesh visualization
- `--include_original` (default: True): Include original video frames
- `--black_background` (default: False): Use black background instead of original
- `--cat_dim` (default: 0): Concatenate videos vertically (0) or horizontally (1)
- `--use_mask` (default: True): Use face mask for visualization

### Face Detection Settings

#### Face Detector
- `face_detector="fan3d"` in the code
- Controls which face detector to use for locating faces in frames

#### Detection Threshold
- Default is 0.5-0.9 depending on the detector
- Higher values create more precise but potentially fewer detections

#### Image Size and Scale
- `image_size=224` (typical default)
- `scale=1.25` (typical default) - controls how much context around the face to include

### Other Configuration Options

#### Input/Output Paths
- `--input_video`: Path to the video file
- `--output_folder`: Directory for results (default: "video_output")
- `--path_to_models`: Directory containing model checkpoints

#### Processing Resume
- `--processed_subfolder`: Resume from previously processed video frames
- Useful for continuing interrupted operations

#### Logging
- `--logger`: Logging method (empty or "wandb")

### Tips for Customization

1. For quality adjustment:
   - Change `mode = 'coarse'` to `mode = 'detail'` in the code for higher quality reconstructions
   - Select different models with the `--model_name` parameter

2. For GPU utilization:
   - Adjust `batch_size` and `num_workers` parameters based on your hardware
   - For multi-GPU setups, the code would need to be modified to use PyTorch's distributed data parallel capabilities

3. For output customization:
   - Use the appropriate command-line flags (`--save_mesh`, `--save_codes`)
   - Adjust visualization options using the `--include_*` parameters and `--cat_dim`

4. For processing large videos:
   - Consider using the `--processed_subfolder` option to save intermediate results
   - Reduce batch size if you encounter memory issues

