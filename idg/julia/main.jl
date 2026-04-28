module IDG

using HDF5

fid = HDF5.h5open("input/input.hdf5", "r")

uvw_dataset = fid["uvws"]
uvws = read(uvw_dataset)

metadata_dataset = fid["metadata"]
metadata = read(metadata_dataset)

close(fid)

end
