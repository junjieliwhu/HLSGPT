# v4, change end_date to start_date, split many subsquences when using model_hls.predict
import os
import numpy as np
import traceback
import calendar
import transformer_encoder44
import pandas as pd
from config import MAX_LANDSAT, MAX_SENTINEL2, BANDS_N,FILL
import rasterio
import tensorflow as tf
from datetime import datetime, timedelta
import sys
from pathlib import Path
import re
import HLS_io_chunks
import gc
np.set_printoptions(suppress=True)

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("TF sees GPUs:", tf.config.list_physical_devices('GPU'))

def search_doy(target_doy, doy_list):
    find = None
    for i in range(len(doy_list)):
        if abs(target_doy - doy_list[i]) < 1e-4:
            find = i
            break
    return find

def find_file(date_str, tile_id, HLS_1year_files):
    l8_path, s2_path = None, None
    for f in HLS_1year_files:
        if f.endswith('.Fmask.tif') and f'HLS.L30.T{tile_id}.{date_str}' in f:
            l8_path = f
        if f.endswith('.Fmask.tif') and f'HLS.S30.T{tile_id}.{date_str}' in f:
            s2_path = f
    return l8_path, s2_path


## get file names one year BEFORE the date time_str
def get_files_1year(tile, time_str, file_list):
    # Convert the provided day_of_year (format: yyyyddd) to a datetime object

    defined_date = datetime.strptime(time_str, "%Y%j")
    end_date = defined_date + timedelta(days=365)  # One year before

    # Build a regex to extract the date from filenames matching the given tile.
    # e.g., for tile T15TUE, a file name may contain "HLS.S30.T15TUE.2023244T170859..."
    pattern = re.compile(r"HLS\.[SL]30\." + re.escape(tile) + r"\.(\d{7})T\d{6}")

    selected_files = []
    seen_basenames = set()

    for fname in file_list:
        if tile in fname:
            m = pattern.search(fname)
            if m:
                # print
                file_date_str = m.group(1)  # e.g., "2023244"
                file_date = datetime.strptime(file_date_str, "%Y%j")
                if defined_date <= file_date < end_date:
                    base_name = os.path.basename(fname)
                    if base_name not in seen_basenames:
                        seen_basenames.add(base_name)
                        selected_files.append(fname)

    selected_files.sort()
    return selected_files, len(selected_files)


def process_patch_yearly(tile_id, HLS_1year_files, row_top, col_left, width, height, reconstructed_dates, is_evaluation, all_dates):
    skipped_days = reconstructed_dates if is_evaluation else []
    periods = len(all_dates)
    patch_data = np.full([height, width, (periods + periods), BANDS_N], fill_value=FILL, dtype=np.float32)
    patch_qa = np.full([height, width, (periods + periods)], fill_value=False, dtype=bool)
    min_year = int(all_dates[0][:4])
    delete_time_index = []
    for i in range(periods):
        date_str = all_dates[i]
        l8_path, s2_path = find_file(date_str, tile_id, HLS_1year_files)
        fyear = int(date_str[:4])
        fdoy = int(date_str[4:])
        patch_data[:, :, i, 0] = int(fyear - min_year) + int(fdoy - 1) / 366.0  ## for DOY add to all the pixels
        patch_data[:, :, i + periods, 0] = int(fyear - min_year) + int(fdoy - 1) / 366.0  ## for DOY add to all the pixels
        if date_str in skipped_days:
            continue
        l8_valid_sum = 0
        if l8_path is not None:
            HLSi = HLS_io_chunks.HLS_tile(l8_path.rstrip("\n"), is_L8=True)
            HLSi.load_data(row_top, col_left, width=width, height=height)
            l8_valid_sum = HLSi.is_valid.sum()
            patch_qa[HLSi.is_valid, i] = True
            for bi in range(1, (BANDS_N - 4)):  ##[1,BANDS_N-4] ## 01-08th BANDS
                patch_data[HLSi.is_valid, i, bi] = HLSi.reflectance_30m[bi - 1, HLSi.is_valid]
        s2_valid_sum = 0
        if s2_path is not None:
            HLSi = HLS_io_chunks.HLS_tile(s2_path.rstrip("\n"), is_L8=False)
            HLSi.load_data(row_top, col_left, width=width, height=height)
            s2_valid_sum = HLSi.is_valid.sum()
            patch_qa[HLSi.is_valid, i + periods] = True
            for bi in range(1, BANDS_N):  ##[1,BANDS_N-1]  ## 01-11th BANDS
                patch_data[HLSi.is_valid, i + periods, bi] = HLSi.reflectance_30m[bi - 1, HLSi.is_valid]

        total_valid_sum = l8_valid_sum + s2_valid_sum
        # if l8, s2 do not contain valid pixels and current doy is not in reconstructed_dates, this time should be removed from patch_data to reduce the periods !!
        if total_valid_sum == 0 and date_str not in reconstructed_dates:
            delete_time_index.extend([i, i + periods])
    if len(delete_time_index) > 0:
        del_idx = np.unique(np.asarray(delete_time_index, dtype=int))
        H, W, T, C = patch_data.shape
        keep_mask = np.ones(T, dtype=bool)
        keep_mask[del_idx] = False
        patch_data_new = patch_data[:, :, keep_mask, :]
        patch_qa_new = patch_qa[:, :, keep_mask]
        return patch_data_new, patch_qa_new

    return patch_data, patch_qa


def norm_data(data, x_mean, x_std, indices, slice_range, offset):
    for index in indices:
        valid_indices = data[:, slice_range, index] != -9999.0
        data[:, slice_range, index][valid_indices] -= x_mean[index + offset]
        data[:, slice_range, index][valid_indices] /= x_std[index + offset]

def load_model(model_path, periods):
    model_basic = transformer_encoder44.get_transformer_reflectance(MAX_LANDSAT=176, MAX_SENTINEL2=176, L8_bands_n=8,
                                                                  S2_bands_n=12,
                                                                  layern1=3, layern2=4, units=256,
                                                                  n_head=8, drop=0.1, is_day_input=1,
                                                                  is_sensor=True, is_xy=False, active="sigmoid",
                                                                  concat=4)
    model_basic.load_weights(model_path)
    if periods == MAX_LANDSAT:
        return model_basic
    model_long = transformer_encoder44.get_transformer_reflectance(MAX_LANDSAT=periods, MAX_SENTINEL2=periods, L8_bands_n=8,
                                                                  S2_bands_n=12,
                                                                  layern1=3, layern2=4, units=256,
                                                                  n_head=8, drop=0.1, is_day_input=1,
                                                                  is_sensor=True, is_xy=False, active="sigmoid",
                                                                  concat=4)
    embedding_name = ""
    for il, ilayer in enumerate(model_basic.layers):
        ilayer1 = model_basic.layers[il]
        ilayer2 = model_long.layers[il]
        # if (model_drop==0 and 'dropout' not in ilayer2.name) or model_drop>0: # to handle one model has dropout while the other does no
        # il1=il1+1
        # else:
        # continue
        # ilayer1 = model    .layers[il1]
        name_cls = ''.join([ic for ic in ilayer1.name if not ic.isdigit() and ic != '_'])
        name_ref = ''.join([ic for ic in ilayer2.name if not ic.isdigit() and ic != '_'])
        if "embedding" in name_cls:
            embedding_name = ilayer1.name
        if name_cls == name_ref and ilayer1.trainable and ilayer2.trainable and not not ilayer1.weights and not not ilayer2.weights:
            # print ("\t"+ilayer.name, end=" ")
            model_long.layers[il].set_weights(model_basic.layers[il].get_weights())
    print('using long model...')
    return model_long


def save_result_as_geotiff(tile, data, output_path):
    tile_example = get_tile_example_img(HLS_LIST, tile)
    with rasterio.open(tile_example) as src:
        profile = src.profile.copy()
        profile.update({
            'count': data.shape[0],
            'dtype': 'float32',
            'nodata': -9999,
            'compress': 'deflate'
        })
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
    print('saved to', output_path)


def get_tile_example_img(HLS_LIST, tile):
    find_file = False
    for line in HLS_LIST:
        if tile in line:
            find_file = True
            return line.strip()
    if not find_file:
        raise FileNotFoundError('No tile example image in ' + HLS_LIST)

def extract_by_doy_optimized(data, target_doys):
    N, T, B = data.shape
    M = len(target_doys)
    doy_sequence = data[0, :, 0]
    target_indices = []
    for doy in target_doys:
        idx = search_doy(doy, doy_sequence.flatten().tolist())
        if idx is not None:
            target_indices.append(idx)
        else:
            target_indices.append(-1)
    result = np.full((N, M, B-1), -9999, dtype=np.float32)
    for m, idx in enumerate(target_indices):
        if idx >= 0:
            result[:, m, :] = data[:, idx, 1:]
    return result

def get_all_dates(HLS_1year_files):
    pattern = re.compile(r"HLS\.[SL]30\.[^\.]+\.(\d{7})T\d{6}")
    date_list = []
    for fname in HLS_1year_files:
        m = pattern.search(fname)
        if m:
            date_str = m.group(1)  # e.g., "2023244"
            date_list.append(date_str)
    return date_list

def process_by_chunks(tile_id, IMG_WIDTH, IMG_HEIGHT, CHUNK_SIZE, reconstructed_dates, start_date, is_evaluation):

    output_landsat = np.full((IMG_HEIGHT, IMG_WIDTH, len(reconstructed_dates), BANDS_N-1), FILL, dtype=np.float32)
    output_sentinel = np.full((IMG_HEIGHT, IMG_WIDTH, len(reconstructed_dates), BANDS_N-1), FILL, dtype=np.float32)

    HLS_1year_files, total_n = get_files_1year("T" + tile_id, str(start_date), file_list=HLS_LIST)

    # get all dates (year+doy)
    files_dates = get_all_dates(HLS_1year_files)
    unique_dates = list(set(files_dates))
    not_includes_dates = [x for x in reconstructed_dates if x not in unique_dates]
    all_dates = sorted(unique_dates + not_includes_dates)
    min_year = int(all_dates[0][:4])
    for row_start in range(0, IMG_HEIGHT, CHUNK_SIZE):
        for col_start in range(0, IMG_WIDTH, CHUNK_SIZE):
            print('start to process row {}, col {}'.format(row_start, col_start))
            row_end = min(row_start + CHUNK_SIZE, IMG_HEIGHT)
            col_end = min(col_start + CHUNK_SIZE, IMG_WIDTH)
            width = col_end - col_start
            height = row_end - row_start
            patch_data, patch_qa = process_patch_yearly(tile_id, HLS_1year_files, row_start, col_start, width, height, reconstructed_dates, is_evaluation, all_dates)    # shape: width, height, MAX_LANDSAT + MAX_SENTINEL2, BANDS (DOY + band) (512, 512, 352, 12)
            periods = int(patch_qa.shape[2] / 2)
            # only input valid pixels to model, num l8>N and s2>N
            # patch_qa_landsat = patch_qa[:, :, :periods]
            # patch_qa_sentinel = patch_qa[:, :, periods:]
            # valid_patch_qa = np.logical_and(patch_qa_landsat.sum(axis=2) > VALID_DATA_THRESHOLD_IN_YEAR, patch_qa_sentinel.sum(axis=2) > VALID_DATA_THRESHOLD_IN_YEAR)
            valid_patch_qa = patch_qa.sum(axis=2) > VALID_DATA_THRESHOLD_IN_YEAR
            qa_flat = valid_patch_qa.reshape(-1) # height * width
            valid_indices = np.where(qa_flat)[0]
            patch_data_reshaped = patch_data.reshape(-1, periods+periods, BANDS_N)
            valid_patches = patch_data_reshaped[valid_indices]  # shape: (N_valid, 352, BANDS=12)
            norm_data(valid_patches, x_mean, x_std, range(1, BANDS_N - 4), slice(0, periods), offset=0)
            norm_data(valid_patches, x_mean, x_std, range(1, BANDS_N), slice(periods, periods + periods), offset=8)
            valid_patches = valid_patches.astype(np.float32, copy=False)
            print('finish preparing chunk data-------------------')

            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model_hls = load_model(hls_transformer_model_path, periods)
            N, T, B = valid_patches.shape
            STEP = int(1e5)
            if N < STEP:
                predictions = model_hls.predict(valid_patches, verbose=2, batch_size=BATCH_SIZE)  # N_valid * 352 * 11
            else:
                predictions = np.full(shape=(N, T, B - 1), fill_value=FILL,
                                      dtype=np.float16)
                for i in range(0, N, STEP):
                    start = i
                    end = min(i + STEP, N)
                    print('subsquence {} to {}'.format(start, end))
                    tempx = valid_patches[start:end]
                    tempy = model_hls.predict(tempx, batch_size=BATCH_SIZE, verbose=2)
                    predictions[start:end] = tempy

            landsat_pred = predictions[:, :periods, :]   # shape: (N_valid, 176, 11)
            sentinel_pred = predictions[:, periods:, :]  # shape: (N_valid, 176, 11)

            print('finish reconstruction-------------------')
            # observations
            landsat_obs = patch_data_reshaped[valid_indices, :periods, :]  # shape: (N_valid, 176, 12)
            sentinel_obs = patch_data_reshaped[valid_indices, periods:, :]  # shape: (N_valid, 176, 12)

            landsat_missing = landsat_obs[:, :, 1] == FILL   #  shape: (N_valid, 176)
            sentinel_missing = sentinel_obs[:, :, 1] == FILL

            landsat_obs[landsat_missing, 1:] = landsat_pred[landsat_missing]
            sentinel_obs[sentinel_missing, 1:] = sentinel_pred[sentinel_missing]

            # Init outputs
            predict_doys = [int(date[:4]) - min_year + (int(date[4:]) - 1) / 366.0 for date in reconstructed_dates]
            result_landsat = np.full((height * width, len(predict_doys), BANDS_N - 1), FILL, dtype=np.float32)
            result_sentinel = np.full((height * width, len(predict_doys), BANDS_N - 1), FILL, dtype=np.float32)

            landsat = extract_by_doy_optimized(landsat_obs, predict_doys)
            sentinel = extract_by_doy_optimized(sentinel_obs, predict_doys)
            result_landsat[valid_indices] = landsat
            result_sentinel[valid_indices] = sentinel

            # Reshape to patch size
            result_landsat = result_landsat.reshape(height, width, len(predict_doys), BANDS_N - 1)
            result_sentinel = result_sentinel.reshape(height, width, len(predict_doys), BANDS_N - 1)

            output_landsat[row_start:row_end, col_start:col_end, :, :] = result_landsat
            output_sentinel[row_start:row_end, col_start:col_end, :, :] = result_sentinel


            tf.keras.backend.clear_session()
            gc.collect()

    return output_landsat, output_sentinel


def days_in_year(year):
    return 366 if calendar.isleap(year) else 365


if __name__ == "__main__":

    #### input parameters#################################
    tile_id = sys.argv[1]  # '14TNP'
    reconstructed_dates = sys.argv[2].split(',')  # string, year+doy, e.g. '2023140', if there are multiple dates, separate them with commas.    '2023039,2023040,2023041'
    start_date = sys.argv[3] # year+doy, the start date of the annual input time series
    hls_data_dir = sys.argv[4] # The input HLS time series dir
    hls_transformer_model_path = sys.argv[5] # model path
    output_dir = sys.argv[6] # output dir
    ######################################################
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #  Default parameters
    mean_std_file = 'mean_std_v1_6_filtered.csv'
    is_evaluation = False  # Default is False, if True, all pixels of the tile on the predicted date will be masked and not participate in model inference, is used to evaluate model reconstruction accuracy
    CHUNK_SIZE = 1220   # 1220, 915, 732
    VALID_DATA_THRESHOLD_IN_YEAR = 4
    IMG_WIDTH = IMG_HEIGHT = 3660
    BATCH_SIZE = 1024

    start = datetime.now()
    print_str = '\n\n\nstart time: ' + str(start)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    search_dir = Path(hls_data_dir)

    HLS_LIST = [str(p) for p in search_dir.rglob('*.Fmask.tif')]
    df = pd.read_csv(mean_std_file, skiprows=1, header=None, names=['mean', 'std'])
    x_mean = df['mean'].values
    x_std = df['std'].values

    output_landsat, output_sentinel = process_by_chunks(tile_id, IMG_WIDTH, IMG_HEIGHT, CHUNK_SIZE, reconstructed_dates, start_date, is_evaluation)

    for i in range(len(reconstructed_dates)):

        landsat_data = output_landsat[:, :, i, :7]  # shape: (H, W, B)
        landsat_data = landsat_data.transpose(2, 0, 1)  # (B, H, W)

        output_path = os.path.join(output_dir, f'tile_{tile_id}_date_{reconstructed_dates[i]}_landsat.tif')
        if os.path.exists(output_path):
            os.remove(output_path)
        save_result_as_geotiff(tile_id, landsat_data, output_path)

        # Sentinel
        sentinel_data = output_sentinel[:, :, i, :]  # shape: (H, W, B)
        sentinel_data = sentinel_data.transpose(2, 0, 1)  # (B, H, W)

        output_path = os.path.join(output_dir, f'tile_{tile_id}_date_{reconstructed_dates[i]}_sentinel.tif')
        if os.path.exists(output_path):
            os.remove(output_path)
        save_result_as_geotiff(tile_id, sentinel_data, output_path)

    end = datetime.now()
    elapsed = end - start
    print_str = '\nEnd time = ' + str(end) + 'Elapsed time = ' + str(
        elapsed) + '\n======================================'
    print(print_str)


