use std::io;

use ndarray_npy::ReadDataError;
use py_literal::Value;

mod frequency;
mod grid;
mod metadata;
mod subgrid;
mod taper;
mod uvw;
mod visibility;
mod wavenumber;

pub use frequency::*;
pub use grid::*;
pub use metadata::*;
pub use subgrid::*;
pub use taper::*;
pub use uvw::*;
pub use visibility::*;
pub use wavenumber::*;

fn check_type_desc(type_desc: &Value, expected: &str) -> Result<(), ReadDataError> {
    // Is this cheating? Potentially.
    // Does it really matter in this context? I don't think so.
    let signature = format!("{}", type_desc);

    if signature == expected {
        Ok(())
    } else {
        Err(ReadDataError::WrongDescriptor(type_desc.clone()))
    }
}

/// Returns `Ok(_)` iff the `reader` had no more bytes on entry to this
/// function.
///
/// This function is taken from `ndarray-npy`, file `src/npy/elements/mod.rs`.
///
/// **Warning** This will consume the remainder of the reader.
fn check_for_extra_bytes<R: io::Read>(reader: &mut R) -> Result<(), ReadDataError> {
    let num_extra_bytes = reader.read_to_end(&mut Vec::new())?;
    if num_extra_bytes == 0 {
        Ok(())
    } else {
        Err(ReadDataError::ExtraBytes(num_extra_bytes))
    }
}
