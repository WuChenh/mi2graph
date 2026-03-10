use chrono::Local;
use ndarray::prelude::*;
use ndarray_npy::NpzWriter;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::path::Path;

mod mutualinfo;
mod processing;
mod slidingwindow;
mod sortf64;

use mutualinfo::iter_feat_pairs_mi;
use processing::{remove_feat_similar, rm_feat_low_cv};
use slidingwindow::init_windows_from_ratio;
use sortf64::get_sort_indices_vecf64;

/// Configuration for sliding window parameters.
#[derive(Debug, Clone, Copy)]
pub struct WindowConfig {
    /// Minimum ratio of window size to the number of samples
    pub ratio_min: f64,
    /// Maximum ratio of window size to the number of samples
    pub ratio_max: f64,
    /// Step size ratio for window size
    pub ratio_step: f64,
    /// Step size ratio for sliding
    pub ratio_slide: f64,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            ratio_min: 0.33,
            ratio_max: 0.98,
            ratio_step: 0.07,
            ratio_slide: 0.07,
        }
    }
}

/// Configuration for feature filtering thresholds.
#[derive(Debug, Clone, Copy)]
pub struct FilterConfig {
    /// Threshold for removing features with low coefficient of variation
    pub thre_cv: f64,
    /// Threshold for removing redundant features (Pearson correlation)
    pub thre_pcc: f64,
    /// Threshold for removing feature pairs with low mutual information
    pub thre_mi: f64,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            thre_cv: 0.1,
            thre_pcc: 0.95,
            thre_mi: 0.05,
        }
    }
}

/// Processing options.
#[derive(Debug, Clone, Copy)]
pub struct ProcessingOptions {
    /// Whether to detect similar features pairs and remove redundant features
    pub check_sim: bool,
    /// Number of top-CV features to retain (0 = use `thre_cv` instead)
    pub n_features_to_select: usize,
    /// Number of threads (0 = use all available - 1)
    pub n_threads: usize,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            check_sim: false,
            n_features_to_select: 5000,
            n_threads: 0,
        }
    }
}

/// Result type for reading parquet files.
pub type ParquetData = (Array2<f64>, DataFrame, Vec<String>);

/// Data arrays to be saved to NPZ file.
struct NpzData<'a> {
    mi_values: &'a Array1<f64>,
    feat_pairs: &'a Array2<i64>,
    processed_mat: &'a Array2<f64>,
    feat_indices: &'a Array1<i64>,
    simi_feat_pairs: &'a Array2<i64>,
}

/// Generate NMIC relations between features with dynamic feature filtering for the next graph initialization.
///
/// **Steps**:
/// 1. Remove features with low coefficients of variation (using dynamic sliding windows).
/// 2. Detect similar features pairs (optional) using dynamic 2D sliding windows for maximizing PCC (`abs=true`) then remove redundant features.
/// 3. Compute NMIC for each feature pair (using dynamic sliding windows).
/// 4. Filter out weak NMIC values and corresponding feature pairs.
/// 5. Save sorted NMIC values, feature pairs, processed input data, feature indices, similar feature pairs and input arguments.
///
/// **Input**:
/// + `path_output`: path to save the result files
/// + `data`: input data with shape (`n_var`, `n_obs`)
/// + `obs_names`: observation names
/// + `var_names`: variable names
/// + `check_sim`: whether to detect similar features pairs and remove redundant features
/// + `n_features_to_select`: number of top-CV features to retain (0 = use `thre_cv` instead)
/// + `filter_config`: thresholds for filtering features and edges
/// + `window_config`: sliding window parameters
/// + `n_threads`: number of threads
///
/// **Output**:
/// + NPZ file
///     + `mi_values`: sorted mutual information values
///     + `feat_pairs`: corresponding feature pairs (edges)
///     + `processed_mat`: The processed input matrix after *Step 3*
///     + `mat_feat_indices`: The feature indices of the processed matrix
///     + `mat_simi_feat_pairs`: Similar feature pairs given by *Step 3*
///     + `input_args`: The input arguments
/// + Parquet file
///     + The named dataframe of `processed_mat`
///
/// Setup the thread pool with the specified number of threads.
fn setup_thread_pool(options: &ProcessingOptions) {
    let mut num_threads = std::thread::available_parallelism().unwrap().get();
    if num_threads > 2 {
        num_threads -= 1;
    }
    if num_threads > options.n_threads && options.n_threads > 0 {
        num_threads = options.n_threads;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    println!("\n⚡️  Using {num_threads} threads");
}

/// Pre-compute the sort indices for each feature.
fn compute_sort_indices(data: &Array2<f64>) -> Vec<Vec<usize>> {
    let features: Vec<Vec<f64>> = (0..data.nrows()).map(|i| data.row(i).to_vec()).collect();
    features
        .par_iter()
        .map(|feat| get_sort_indices_vecf64(feat))
        .collect()
}

/// Filter features based on CV threshold.
fn filter_by_cv(
    data: &Array2<f64>,
    filter_config: FilterConfig,
    options: &ProcessingOptions,
    sliding_windows: &[Vec<usize>],
    features_sort_indices: &mut Vec<Vec<usize>>,
) -> (Array2<f64>, Vec<usize>) {
    println!("\nStart feature selection at: {:?}\n", Local::now());

    let (data_filtered, feat_indices, new_sort_indices) = rm_feat_low_cv(
        data,
        filter_config.thre_cv,
        options.n_features_to_select,
        sliding_windows,
        features_sort_indices,
    );
    *features_sort_indices = new_sort_indices;

    println!(
        "✅ Shape of data after removing features with low CV (coefficient of variation) values: {:?} (n_feat x n_obs)",
        data_filtered.shape()
    );

    (data_filtered, feat_indices)
}

/// Remove similar features if enabled.
fn filter_similar_features(
    mut data: Array2<f64>,
    mut feat_indices: Vec<usize>,
    filter_config: FilterConfig,
    options: &ProcessingOptions,
    sliding_windows: &[Vec<usize>],
    features_sort_indices: &mut Vec<Vec<usize>>,
) -> (Array2<f64>, Vec<usize>, Array2<i64>, Vec<Vec<usize>>) {
    let mut simi_feat_pairs = Array2::<i64>::zeros((0, 2));

    if options.check_sim {
        println!("\nStart removing similar features.");
        (data, feat_indices, simi_feat_pairs, *features_sort_indices) = remove_feat_similar(
            &data,
            filter_config.thre_pcc,
            sliding_windows,
            features_sort_indices,
        );
        println!(
            "✅ Shape of data after removing similar features: {:?}",
            data.shape()
        );
    } else {
        println!("\n✅ Skip removing similar features.");
    }

    (
        data,
        feat_indices,
        simi_feat_pairs,
        features_sort_indices.clone(),
    )
}

/// Compute NMIC for all feature pairs.
fn compute_nmic(
    data: &Array2<f64>,
    sliding_windows: &[Vec<usize>],
    features_sort_indices: &[Vec<usize>],
) -> (Array1<f64>, Array2<i64>) {
    let time_start = Local::now();
    println!("\nStart computing NMIC relations: {time_start:?}");

    let result = iter_feat_pairs_mi(data, sliding_windows, features_sort_indices, true);

    let time_end = Local::now();
    println!("End time:   {time_end:?}");
    let time_cost = time_end - time_start;
    let time_cost_seconds = time_cost.num_seconds();
    let time_cost_hours = time_cost_seconds as f64 / 3600.0;
    println!("⏱️ Elapsed time:  {time_cost_seconds} seconds, or {time_cost_hours:.2} hours\n");

    result
}

/// Filter edges with MI below threshold and convert indices.
/// Also updates simi_feat_pairs indices if check_sim is enabled.
fn filter_and_convert_edges(
    mut data: Array2<f64>,
    mi_values: &Array1<f64>,
    feat_pairs: &Array2<i64>,
    feat_indices_1: &[usize],
    feat_indices_0: &[usize],
    thre_mi: f64,
    check_sim: bool,
    mut simi_feat_pairs: Array2<i64>,
) -> (
    Array2<f64>,
    Array1<f64>,
    Array2<i64>,
    Array1<i64>,
    Array2<i64>,
) {
    // Remove low-NMIC feature pairs.
    let last2keep = check_sorted_vals(mi_values, thre_mi);
    let mi_values_o: Array1<f64> = mi_values.slice(s![..last2keep]).to_owned();
    let mut feat_pairs_o: Array2<i64> = feat_pairs.slice(s![..last2keep, ..]).to_owned();
    println!(
        "Number of feature pairs after removing weak NMIC values: {}",
        last2keep + 1
    );

    // Get sorted unique features in pairs
    let flattened: Array1<i64> = feat_pairs_o.iter().copied().collect();
    let uniq_features: HashSet<i64> = flattened.into_iter().collect();
    let mut sorted_uniq_features: Vec<usize> = uniq_features.iter().map(|&x| x as usize).collect();
    sorted_uniq_features.sort_unstable();

    // Convert feat_pairs_o based on feat_indices_1
    let map_1: HashMap<usize, usize> = sorted_uniq_features
        .iter()
        .map(|&i| (i, feat_indices_1[i]))
        .collect();

    feat_pairs_o.outer_iter_mut().for_each(|mut row| {
        row[0] = map_1[&(row[0] as usize)] as i64;
        row[1] = map_1[&(row[1] as usize)] as i64;
    });

    // Remove features of data that are not in sorted_uniq_features
    data = data.select(Axis(0), &sorted_uniq_features);

    // Convert features indices after NMIC filtering
    sorted_uniq_features.par_iter_mut().for_each(|i| {
        *i = map_1[i];
    });

    // Convert feature indices to original indices.
    if check_sim {
        let map_new2orig: HashMap<usize, usize> = sorted_uniq_features
            .iter()
            .map(|&i| (i, feat_indices_0[i]))
            .collect();

        feat_pairs_o.outer_iter_mut().for_each(|mut row| {
            row[0] = map_new2orig[&(row[0] as usize)] as i64;
            row[1] = map_new2orig[&(row[1] as usize)] as i64;
        });
        sorted_uniq_features.par_iter_mut().for_each(|i| {
            *i = map_new2orig[i];
        });
        // Also convert simi_feat_pairs indices
        simi_feat_pairs.outer_iter_mut().for_each(|mut row| {
            row[0] = map_new2orig[&(row[0] as usize)] as i64;
            row[1] = map_new2orig[&(row[1] as usize)] as i64;
        });
    }

    println!("Number of features: {}", sorted_uniq_features.len());

    let data_1_feat_indices_o: Array1<i64> =
        Array1::from_vec(sorted_uniq_features.par_iter().map(|&i| i as i64).collect());

    (
        data,
        mi_values_o,
        feat_pairs_o,
        data_1_feat_indices_o,
        simi_feat_pairs,
    )
}

/// Generate NMIC relations between features with dynamic feature filtering for the next graph initialization.
///
/// **Steps**:
/// 1. Remove features with low coefficients of variation (using dynamic sliding windows).
/// 2. Detect similar features pairs (optional) using dynamic 2D sliding windows for maximizing PCC (`abs=true`) then remove redundant features.
/// 3. Compute NMIC for each feature pair (using dynamic sliding windows).
/// 4. Filter out weak NMIC values and corresponding feature pairs.
/// 5. Save sorted NMIC values, feature pairs, processed input data, feature indices, similar feature pairs and input arguments.
///
/// **Input**:
/// + `path_output`: path to save the result files
/// + `data`: input data with shape (`n_var`, `n_obs`)
/// + `obs_names`: observation names
/// + `var_names`: variable names
/// + `check_sim`: whether to detect similar features pairs and remove redundant features
/// + `n_features_to_select`: number of top-CV features to retain (0 = use `thre_cv` instead)
/// + `filter_config`: thresholds for filtering features and edges
/// + `window_config`: sliding window parameters
/// + `n_threads`: number of threads
///
/// **Output**:
/// + NPZ file
///     + `mi_values`: sorted mutual information values
///     + `feat_pairs`: corresponding feature pairs (edges)
///     + `processed_mat`: The processed input matrix after *Step 3*
///     + `mat_feat_indices`: The feature indices of the processed matrix
///     + `mat_simi_feat_pairs`: Similar feature pairs given by *Step 3*
///     + `input_args`: The input arguments
/// + Parquet file
///     + The named dataframe of `processed_mat`
///
/// # Panics
///
/// This function may panic if:
/// - Thread pool initialization fails
/// - File operations fail
/// - Internal unwrap operations fail (this should be improved in future versions)
pub fn mic_mat_with_data_filter(
    path_output: &str,
    data: &Array2<f64>,
    obs_names: &DataFrame,
    var_names: &[String],
    filter_config: FilterConfig,
    window_config: WindowConfig,
    options: ProcessingOptions,
) -> Result<(), Box<dyn Error>> {
    // 1. Setup thread pool
    setup_thread_pool(&options);

    // 2. Initialize sliding windows
    let sliding_windows = init_windows_from_ratio(
        data.ncols(),
        window_config.ratio_min,
        window_config.ratio_max,
        window_config.ratio_step,
        window_config.ratio_slide,
    );

    // 3. Pre-compute sort indices
    let mut features_sort_indices = compute_sort_indices(data);

    // 4. Filter by CV
    let (data_filtered, feat_indices_0) = filter_by_cv(
        data,
        filter_config,
        &options,
        &sliding_windows,
        &mut features_sort_indices,
    );

    // 5. Filter similar features
    let (data_0, feat_indices_1, simi_feat_pairs, features_sort_indices) = filter_similar_features(
        data_filtered,
        feat_indices_0.clone(),
        filter_config,
        &options,
        &sliding_windows,
        &mut features_sort_indices,
    );

    // 6. Compute NMIC
    let (mi_values, feat_pairs) = compute_nmic(&data_0, &sliding_windows, &features_sort_indices);

    // 7. Filter edges and convert indices
    let (data_final, mi_values_o, feat_pairs_o, data_1_feat_indices_o, simi_feat_pairs) =
        filter_and_convert_edges(
            data_0,
            &mi_values,
            &feat_pairs,
            &feat_indices_1,
            &feat_indices_0,
            filter_config.thre_mi,
            options.check_sim,
            simi_feat_pairs,
        );

    // 8. Save results
    let npz_data = NpzData {
        mi_values: &mi_values_o,
        feat_pairs: &feat_pairs_o,
        processed_mat: &data_final,
        feat_indices: &data_1_feat_indices_o,
        simi_feat_pairs: &simi_feat_pairs,
    };
    save_npz(path_output, npz_data, filter_config, window_config)?;

    save_parquet(
        &format!("{path_output}.parquet"),
        &data_final,
        obs_names,
        var_names,
        &data_1_feat_indices_o
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>(),
    )?;

    Ok(())
}

/// Save all data to a NPZ file
fn save_npz(
    path_output_npz: &str,
    data: NpzData<'_>,
    filter_config: FilterConfig,
    window_config: WindowConfig,
) -> Result<(), Box<dyn Error>> {
    let mut path_npz = format!("{path_output_npz}.npz");
    // Check if path_output_npz ends with .npz
    if path_output_npz.ends_with(".npz") {
        path_npz = path_output_npz.to_string();
    }
    // Check if the file exists. If it does, add timestamp to the file name.
    if Path::new(&path_npz).exists() {
        let timestamp = Local::now().format("%Y%m%d%H%M%S");
        if path_output_npz.ends_with(".npz") {
            path_npz = path_output_npz.replace(".npz", &format!("_{timestamp}.npz"));
        } else {
            path_npz = format!("{path_output_npz}_{timestamp}.npz");
        }
    }
    let path_npz_c = &path_npz.clone();

    let file_npz = File::create(path_npz_c)?;
    let mut npz = NpzWriter::new_compressed(file_npz);
    npz.add_array("mi_values", data.mi_values)?;
    npz.add_array("feat_pairs", data.feat_pairs)?;
    npz.add_array("processed_mat", data.processed_mat)?;
    npz.add_array("mat_feat_indices", data.feat_indices)?;
    npz.add_array("mat_simi_feat_pairs", data.simi_feat_pairs)?;
    // Save input parameters as length 1 ndarray
    npz.add_array("thre_cv", &Array1::from_vec(vec![filter_config.thre_cv]))?;
    npz.add_array("thre_pcc", &Array1::from_vec(vec![filter_config.thre_pcc]))?;
    npz.add_array("thre_mi", &Array1::from_vec(vec![filter_config.thre_mi]))?;
    npz.add_array(
        "ratio_max_window",
        &Array1::from_vec(vec![window_config.ratio_max]),
    )?;
    npz.add_array(
        "ratio_min_window",
        &Array1::from_vec(vec![window_config.ratio_min]),
    )?;
    npz.add_array(
        "ratio_step_window",
        &Array1::from_vec(vec![window_config.ratio_step]),
    )?;
    npz.add_array(
        "ratio_step_sliding",
        &Array1::from_vec(vec![window_config.ratio_slide]),
    )?;
    // Finish writing
    npz.finish()?;

    println!("Results have been saved to \"{path_npz_c}\"");

    Ok(())
}

/// Save processed matrix to a parquet file
fn save_parquet(
    path_output_parquet: &str,
    processed_mat: &Array2<f64>,
    obs_names: &DataFrame,
    var_names: &[String],
    feat_indices: &[usize],
) -> Result<(), Box<dyn Error>> {
    // Check if path_output_parquet ends with .parquet
    let mut path_parquet = format!("{path_output_parquet}.parquet");
    if path_output_parquet.ends_with(".parquet") {
        path_parquet = path_output_parquet.to_string();
    }
    // Check if the file exists. If it does, add timestamp to the file name.
    if Path::new(&path_parquet).exists() {
        let timestamp = Local::now().format("%Y%m%d%H%M%S");
        if path_output_parquet.ends_with(".parquet") {
            path_parquet =
                path_output_parquet.replace(".parquet", &format!("_{timestamp}.parquet"));
        } else {
            path_parquet = format!("{path_output_parquet}_{timestamp}.parquet");
        }
    }
    let path_parquet_c = &path_parquet.clone();

    // Pick saved var (feature) names based on feat_indices
    let saved_var_names = var_names
        .iter()
        .enumerate()
        .filter(|(i, _)| feat_indices.contains(i))
        .map(|(_, x)| x.clone())
        .collect::<Vec<_>>();

    // Create a new DataFrame with processed_mat, obs_names and the selected var_names
    let vec_columns: Vec<Column> = saved_var_names
        .iter()
        .zip(processed_mat.outer_iter())
        .map(|(name, values)| Column::new(name.into(), values.to_vec()))
        .collect();

    let mut df1 = obs_names.hstack(&vec_columns)?;

    let mut file = File::create(path_parquet_c)?;
    ParquetWriter::new(&mut file).finish(&mut df1)?;

    println!(
        "Processed matrix has been saved as a polars dataframe containing observation names and feature names: \"{path_parquet_c}\""
    );
    Ok(())
}

/// Read parquet file (`n_obs` x (1 + `n_vars`)) into ndarray (`n_obs` x `n_vars`)
pub fn read_parquet_to_array2d(path_parquet: &str) -> Result<ParquetData, Box<dyn Error>> {
    let mut file = std::fs::File::open(path_parquet).unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();
    // let mat = df.to_ndarray::<f64>().unwrap();
    // let lf1 = LazyFrame::scan_parquet(path_parquet, Default::default())?;
    let obs_names = df.select(["obs_names"])?;
    let binding = df.drop("obs_names")?;
    let var_names_0 = binding.get_column_names();
    // Convert Vec<&PlSmallStr> to Vec<String>
    let var_names: Vec<String> = var_names_0
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let mat = binding
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap()
        .t()
        .to_owned();
    Ok((mat, obs_names, var_names))
}

/// `mi_values` and `feat_pairs` have been sorted. We can check elements from the end.
fn check_sorted_vals(vals: &Array1<f64>, threshold: f64) -> usize {
    let num_pairs = vals.len();
    // Start from the end
    let mut chk_i = num_pairs - 1;
    while chk_i > 0 {
        // Stop if the current pair has a higher MI than the threshold
        if vals[chk_i] > threshold {
            break;
        }
        chk_i -= 1;
    }
    chk_i
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_window_config_default() {
        let config = WindowConfig::default();
        assert_eq!(config.ratio_min, 0.33);
        assert_eq!(config.ratio_max, 0.98);
        assert_eq!(config.ratio_step, 0.07);
        assert_eq!(config.ratio_slide, 0.07);
    }

    #[test]
    fn test_filter_config_default() {
        let config = FilterConfig::default();
        assert_eq!(config.thre_cv, 0.1);
        assert_eq!(config.thre_pcc, 0.95);
        assert_eq!(config.thre_mi, 0.05);
    }

    #[test]
    fn test_processing_options_default() {
        let options = ProcessingOptions::default();
        assert!(!options.check_sim);
        assert_eq!(options.n_features_to_select, 5000);
        assert_eq!(options.n_threads, 0);
    }

    #[test]
    fn test_check_sorted_vals() {
        let vals = Array1::from_vec(vec![0.01, 0.02, 0.03, 0.04, 0.05]);

        // Threshold below all values, should return last index
        let result = check_sorted_vals(&vals, 0.0);
        assert_eq!(result, 4);

        // Threshold above some values
        let result = check_sorted_vals(&vals, 0.035);
        assert_eq!(result, 3);

        // Threshold above all values
        let result = check_sorted_vals(&vals, 0.06);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_slidingwindow_init() {
        use crate::slidingwindow::init_windows_from_ratio;

        let windows = init_windows_from_ratio(100, 0.33, 0.5, 0.1, 0.1);

        // Should create windows successfully
        assert!(!windows.is_empty());

        // Each window should have start and end
        for window in &windows {
            assert_eq!(window.len(), 2);
            assert!(window[0] < window[1]);
        }
    }

    #[test]
    fn test_sortf64() {
        use crate::sortf64::{get_sort_indices_vecf64, sort_vec_f64};

        let vec = vec![3.0, 1.0, 2.0, 5.0, 4.0];

        // Test sort_vec_f64
        let sorted = sort_vec_f64(&vec);
        assert_eq!(sorted, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Test get_sort_indices_vecf64
        let indices = get_sort_indices_vecf64(&vec);
        assert_eq!(indices, vec![1, 2, 0, 4, 3]);
    }

    #[test]
    fn test_processing_cv_filter() {
        use crate::processing::rm_feat_low_cv;
        use crate::slidingwindow::init_windows_from_ratio;

        // Create test data: 5 features, 10 samples
        let data = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], // Low variation feature
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], // Zero variation feature
        ];

        let sliding_windows = init_windows_from_ratio(10, 0.5, 0.9, 0.2, 0.2);

        // Pre-compute sort indices
        let features_sort_indices: Vec<Vec<usize>> = (0..data.nrows())
            .map(|i| get_sort_indices_vecf64(&data.row(i).to_vec()))
            .collect();

        // Test with n_features_to_select
        let (filtered_data, kept_indices, _) =
            rm_feat_low_cv(&data, 0.1, 3, &sliding_windows, &features_sort_indices);

        assert_eq!(filtered_data.nrows(), 3);
        assert_eq!(kept_indices.len(), 3);
    }
}
