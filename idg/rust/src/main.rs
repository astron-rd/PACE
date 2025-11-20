use std::f64::consts::PI;

mod idgtypes;
mod init;

const NR_CORRELATIONS_IN: usize = 2; // XX, YY
const NR_CORRELATIONS_OUT: usize = 1; // I
const SUBGRID_SIZE: usize = 32; // size of each subgrid
const GRID_SIZE: usize = 1024; // size of the full grid
const OBSERVATION_HOURS: usize = 4; // total observation time in hours
const NR_TIMESTEPS: usize = OBSERVATION_HOURS * 3600;
const NR_CHANNELS: usize = 16; // number of frequency channels
const W_STEP: f64 = 1.0; // w step in wavelengths

const START_FREQUENCY: f64 = 150e6; // 150 MHz
const FREQUENCY_INCREMENT: f64 = 1e6; // 1 MHz
const END_FREQUENCY: f64 = START_FREQUENCY + NR_CHANNELS as f64 * FREQUENCY_INCREMENT;

const SPEED_OF_LIGHT: f64 = 299792458.0;
const IMAGE_SIZE: f64 = SPEED_OF_LIGHT / END_FREQUENCY;

const NR_STATIONS: usize = 20;
const NR_BASELINES: usize = NR_STATIONS * (NR_STATIONS - 1) / 2;

fn main() {
    let _uvw = init::get_uvw(OBSERVATION_HOURS, NR_BASELINES, GRID_SIZE, None, None);
    // TODO: Save uvw to file

    println!("Initialize frequencies");
    let frequencies = init::get_frequencies(START_FREQUENCY, FREQUENCY_INCREMENT, NR_CHANNELS);
    let _wavenumbers = (frequencies * 2. * PI) / SPEED_OF_LIGHT;
    // TODO: Save frequencies to file

    print!("Initialize metadata");
    // TODO: Get metadata and save to file
}
