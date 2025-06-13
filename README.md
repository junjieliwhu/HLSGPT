# HLSGPT
A generative pretrained Transformer for harmonized Landsat and Sentinel-2 reflectance data reconstruction
## Requirements
- **Programming Languages**: Python 3.7+
- **Libraries**:
  - `tensorflow`
  - `numpy`
  - `rasterio`
## Included files
1. `Pro_HLS_GPT_application_v1.py` 
-This is the main script for applying the pretrained HLS model to reconstruct HLS time series.
2. `HLS_io_chunks.py` 
-Class defined for reading HLS tiles.
3. `config.py`
-HLS bands information
4. `read_HLS_time_series.py`
-Functions to read HLS time series
5. `transformer_encoder44.py`
-HLS pretrained model definition
6. `multi_head_from_ChatGPT.py`
-multi head attention function 
7. `mean_std_v1_6_filtered.csv`
-csv file storing the mean and standard deviation values for each band used for normalization
## Usage
`python Pro_HLS_GPT_application_v1.py <tile_id> <YEAR> <END_DOY> <PREDICT_DATES> <HLS_DATA_DIR> <hls_transformer_model_path> <output_dir>`
 - tile_id: The HLS tile name, e.g., '14TNP'.
 - YEAR: The year of input time series, e.g., 2023
 - END_DOY: The end doy (day of year) of input time series, e.g., 365 means the input time series is the whole year of 2023，152 means the input time series is 06/01/2022-06/01/2023  
 - PREDICT_DATES: The dates for which the HLS image is generated, e.g., '2023140', If there are multiple dates, separate them with commas.
 - HLS_DATA_DIR: The input HLS data directory.
 - hls_transformer_model_path: The pretrained Transformer model path.
 - output_dir: The output directory.

## Citation
None
