FROM ubuntu:24.04

ENV CASACORE_VERSION=v3.7.1
ENV EVERYBEAM_VERSION=master

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    bison \
    cmake \
    flex \
    g++ \
    gcc \
    gcovr \
    gfortran \
    git \
    libblas-dev \
    libboost-all-dev \
    libboost-test-dev \
    libboost-python-dev \
    libcfitsio-dev \
    libfftw3-dev \
    libgsl-dev \
    libhdf5-dev \
    liblapack-dev \
    make \
    python3 \
    python3-dev \
    python3-pip \
    wcslib-dev \
    wget &&\
    python3 -m pip install clang-format==14.0.0 cmake-format pre-commit clang-tidy seaborn --break-system-packages
RUN mkdir /src
WORKDIR /src
RUN git clone --single-branch --branch ${CASACORE_VERSION} https://github.com/casacore/casacore/
RUN cmake -B build_casacore casacore -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=/usr/local -DUSE_READLINE=False &&\
    make -j3 -C build_casacore install


RUN git clone --single-branch --branch ${EVERYBEAM_VERSION} https://git.astron.nl/RD/EveryBeam/
RUN cmake -B build_EveryBeam EveryBeam -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=/usr/local &&\
    make -j3 -C build_EveryBeam install

# Install the casacore measures data. We purposely do not install these from
# the Ubuntu repository, but download the latest version directly from the
# ASTRON ftp site.
# Note: The file on the ftp site is updated daily. When warnings regarding
# leap seconds appear, ignore them or regenerate the docker image.
RUN mkdir -p /usr/local/share/casacore/data && \
    wget -qO - https://www.astron.nl/iers/WSRT_Measures.ztar | \
    tar -C /usr/local/share/casacore/data -xzf -


ENV LD_LIBRARY_PATH=/usr/local/lib/