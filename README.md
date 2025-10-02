# HLSGPT
A generative pretrained Transformer for harmonized Landsat and Sentinel-2 reflectance data reconstruction.  
  
HLS-GPT aims to provide a ready-to-use pretrained model that reconstructs Landsat and Sentinel-2 observations for any date across the conterminous United States (CONUS) without imposing temporal shape constraints. 

## Requirements
- **Programming Languages**: Python 3.7+
- **Libraries**:
  - `tensorflow`
  - `numpy`
  - `rasterio`
## Included files
1. `Pro_HLS_GPT_application_v3.py` 
-This is the main script for applying the pretrained HLS model to reconstruct HLS tiles.
2. `HLS_io_chunks.py` 
-Class defined for reading HLS tiles.
3. `config.py`
-Band metadata and constants.
4. `transformer_encoder44.py`
-HLS pretrained model definition
5. `multi_head_from_ChatGPT.py`
-multi head attention function 
6. `mean_std_v1_6_filtered.csv`
-csv file storing the mean and standard deviation values for each band used for normalization
## Usage
```
python Pro_HLS_GPT_application_v3.py \
  <tile_id> <reconstructed_dates> <end_date> <hls_data_dir> \
  <hls_transformer_model_path> <output_dir>
```
### Arguments
 - tile_id: The HLS tile name, e.g., '14TNP'.
 - reconstructed_dates: The dates for which the HLS image is reconstructed, using year+DOY, e.g., '2023140'. If there are multiple dates, separate them with commas.
 - end_date: Define the tail date of the input time series. The model uses observations from [END_DATE - 365, END_DATE]. e.g., '2023365' means the input time series is the whole year of 2023，'2023152' means the input time series is 06/01/2022-06/01/2023  
 - hls_data_dir: The input HLS time series directory.
 - hls_transformer_model_path: The pretrained Transformer model path.
 - output_dir: The output directory.

## Citation
More details can refer to the paper: Li, J., Zhang, H. K., Roy, D. P., and Qiu, Y. (2025). HLS-GPT: A Generative Pretrained Transformer (GPT) Model for Accurate Harmonized Landsat and Sentinel-2 (HLS) Annual Reflectance Time Series Reconstruction. In review.
