use anyhow::Result;

use crate::{
    cli::{Cli, Commands},
    constants::{Float, SPEED_OF_LIGHT},
    types::*,
    util::{print_header, print_param, time_function},
};

pub struct Input {
    pub uvw: UvwArray,
    pub frequencies: FrequencyArray,
    pub wavenumbers: WavenumberArray,
    pub visibilities: VisibilityArray,

    pub metadata: MetadataArray,
    pub taper: Taper,

    pub grid_size: u32,
    pub subgrid_size: u32,
    pub subgrid_count: usize,
    pub image_size: Float,
    pub correlation_count_in: u32,
    pub correlation_count_out: u32,
    pub w_step: Float,
}

impl Input {
    pub fn from_cli(cli: &Cli) -> Result<Self> {
        match &cli.command {
            Commands::Generate {
                subgrid_size,
                grid_size,
                observation_hours,
                channel_count,
                station_count,
                start_frequency,
                frequency_increment,
                ellipticity,
                random_seed,
                point_sources_count,
                max_pixel_offset,
                correlation_count_in,
                correlation_count_out,
                ..
            } => {
                print_header!("GENERATING INPUT DATA");

                let timestep_count = (observation_hours * 3600.0).floor() as u32;
                let baseline_count = (station_count * (station_count - 1)) / 2;

                let uvw: UvwArray = time_function!(
                    "generate uvws",
                    UvwArray::generate(
                        *ellipticity,
                        *random_seed,
                        timestep_count,
                        baseline_count,
                        *grid_size,
                    )
                );

                let frequencies: FrequencyArray = time_function!(
                    "generate frequencies",
                    FrequencyArray::generate(
                        *start_frequency,
                        *channel_count,
                        *frequency_increment
                    )
                );

                let wavenumbers: WavenumberArray = time_function!(
                    "derive wavenumbers",
                    WavenumberArray::from_frequencies(&frequencies)
                );

                let metadata: MetadataArray = time_function!(
                    "generate metadata",
                    MetadataArray::generate(*grid_size, *subgrid_size, *channel_count, &uvw)
                );

                let subgrid_count = metadata.len();

                let end_frequency =
                    start_frequency + ((*channel_count - 1) as Float * frequency_increment);
                let image_size = SPEED_OF_LIGHT / end_frequency;
                let max_pixel_offset = max_pixel_offset.unwrap_or(grid_size / 3);

                let visibilities: VisibilityArray = time_function!(
                    "generate visibilities",
                    VisibilityArray::generate(
                        *point_sources_count,
                        max_pixel_offset,
                        *random_seed,
                        baseline_count,
                        timestep_count,
                        *channel_count,
                        *correlation_count_in,
                        image_size,
                        *grid_size,
                        &frequencies,
                        &uvw,
                    )
                );

                let taper: Taper = time_function!("generate taper", Taper::generate(*subgrid_size));

                Ok(Self {
                    uvw,
                    frequencies,
                    wavenumbers,
                    visibilities,
                    metadata,
                    taper,
                    grid_size: *grid_size,
                    subgrid_size: *subgrid_size,
                    subgrid_count,
                    correlation_count_in: *correlation_count_in,
                    correlation_count_out: *correlation_count_out,
                    w_step: cli.w_step,
                    image_size,
                })
            }
            Commands::Load {
                data_dir,
                uvw_file,
                frequencies_file,
                visibilities_file,
                metadata_file,
                subgrid_size,
                grid_size,
                correlation_count_out,
            } => {
                print_header!("READING INPUT DATA");

                let data_dir = data_dir.clone().unwrap_or(std::env::current_dir()?);

                let uvw: UvwArray =
                    time_function!("load uvws", UvwArray::from_file(&data_dir.join(uvw_file))?);

                let frequencies: FrequencyArray = time_function!(
                    "load frequencies",
                    FrequencyArray::from_file(&data_dir.join(frequencies_file))?
                );

                let wavenumbers: WavenumberArray = time_function!(
                    "derive wavenumbers",
                    WavenumberArray::from_frequencies(&frequencies)
                );

                let visibilities: VisibilityArray = time_function!(
                    "load visibilities",
                    VisibilityArray::from_file(&data_dir.join(visibilities_file))?
                );

                let channel_count = frequencies.shape()[0];

                let metadata: MetadataArray = match metadata_file {
                    None => time_function!(
                        "generate metadata",
                        MetadataArray::generate(
                            *grid_size,
                            *subgrid_size,
                            channel_count as u32,
                            &uvw
                        )
                    ),
                    Some(path) => time_function!(
                        "load metadata",
                        MetadataArray::from_file(&data_dir.join(path))?
                    ),
                };

                let taper: Taper = time_function!("generate taper", Taper::generate(*subgrid_size));

                Ok(Self {
                    subgrid_count: metadata.len(),
                    image_size: SPEED_OF_LIGHT
                        / frequencies.last().expect("frequencies should not be empty"),
                    correlation_count_in: visibilities.shape()[3].try_into().unwrap(),
                    correlation_count_out: *correlation_count_out,
                    w_step: cli.w_step,
                    uvw,
                    frequencies,
                    wavenumbers,
                    visibilities,
                    metadata,
                    taper,
                    grid_size: *grid_size,
                    subgrid_size: *subgrid_size,
                })
            }
        }
    }
}

impl Input {
    pub fn print_parameters(&self) {
        print_header!("PARAMETERS");

        print_param!("correlation_count_in", self.correlation_count_in);
        print_param!("correlation_count_out", self.correlation_count_out);
        print_param!("start_frequency", self.frequencies[0]);
        print_param!(
            "frequency_increment",
            self.frequencies[1] - self.frequencies[0]
        );
        print_param!("nr_channels", self.frequencies.shape()[0]);
        print_param!("nr_timesteps", self.uvw.shape()[1]);
        print_param!("nr_baselines", self.uvw.shape()[0]);
        print_param!("image_size", self.image_size);
        print_param!("subgrid_size", self.subgrid_size);
        print_param!("grid_size", self.grid_size);
    }
}
