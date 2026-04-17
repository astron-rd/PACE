use std::path::PathBuf;

use hdf5_metno::File;
use idg::types::{
    FrequencyArray, FrequencyArrayExtension, MetadataArray, MetadataArrayExtension,
    UvwArrayExtension, VisibilityArray, VisibilityArrayExtension,
};

fn main() -> anyhow::Result<()> {
    println!("Converting input npys into HDF5...");

    let uvws = idg::types::UvwArray::from_npy_file(&PathBuf::from("input/uvw.npy"))?;
    let metadata = MetadataArray::from_npy_file(&PathBuf::from("input/metadata.npy"))?;
    let frequencies = FrequencyArray::from_npy_file(&PathBuf::from("input/frequencies.npy"))?;
    let visibilities = VisibilityArray::from_npy_file(&PathBuf::from("input/visibilities.npy"))?;

    let hdf5_output = File::create("output.hdf5")?;
    let builder = hdf5_output.new_dataset_builder();
    builder.with_data(&uvws).create("uvws")?;
    let builder = hdf5_output.new_dataset_builder();
    builder.with_data(&metadata).create("metadata")?;
    let builder = hdf5_output.new_dataset_builder();
    builder.with_data(&frequencies).create("frequencies")?;
    let builder = hdf5_output.new_dataset_builder();
    builder.with_data(&visibilities).create("visibilities")?;

    Ok(())
}
