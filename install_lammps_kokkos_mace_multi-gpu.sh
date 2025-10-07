#!/bin/bash

# ==============================================================
#  LAMMPS with ML-IAP (MACE + Kokkos Multi-GPU) Installation Script
#  Target system: CSC Mahti GPU partition
#  Usage: ./install_lammps_mliap.sh <username> <project_name>
# ==============================================================

set -e  # Exit on error

# -------------------------------
#  Colors for better readability
# -------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# -------------------------------
#  Check user input
# -------------------------------
if [ $# -lt 2 ]; then
    print_error "Usage: $0 <username> <project_name>"
    exit 1
fi

USERNAME=$1
PROJECT_NAME=$2

print_status "Starting LAMMPS-MLIAP installation for user: $USERNAME (project: $PROJECT_NAME)"

# -------------------------------
#  Set up environment paths
# -------------------------------
export PROJAPPL="/projappl/$PROJECT_NAME/$USERNAME/LAMMPS-MLIAP"
export TMPDIR=${TMPDIR:-/tmp}

print_status "Installation directory: $PROJAPPL"
print_status "Temporary directory: $TMPDIR"

mkdir -p $PROJAPPL

# -------------------------------
#  Load Mahti environment modules
# -------------------------------
print_status "Loading required Mahti modules..."
module purge
module load gcc/11.2.0 openmpi/4.1.2 fftw/3.3.10-mpi cuda/11.5.0 cudnn/8.3.3.40-11.5 intel-oneapi-mkl/2021.4.0
print_success "Modules loaded."

# -------------------------------
#  Python virtual environment
# -------------------------------
print_status "Setting up Python virtual environment for MACE + ML-IAP..."
python3 -m venv $PROJAPPL/venv
source $PROJAPPL/venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install torch==2.4.0 cupy-cuda12x cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12 numpy<2

print_success "Python environment and dependencies installed."

# -------------------------------
#  Get LAMMPS (develop branch)
# -------------------------------
cd $TMPDIR
if [ -d "lammps" ]; then
    print_warning "Removing existing lammps directory..."
    rm -rf lammps
fi

print_status "Cloning official LAMMPS develop branch..."
git clone --branch=develop --depth=1 https://github.com/lammps/lammps
cd lammps
print_success "LAMMPS source downloaded."

# -------------------------------
#  Configure Kokkos preset
# -------------------------------
print_status "Setting up Kokkos GPU preset..."
mkdir -p cmake/presets
cat << EOF > cmake/presets/mahti-gpu.cmake
set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
set(Kokkos_ENABLE_CUDA_UVM ON CACHE BOOL "")
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "")
EOF
print_success "Kokkos GPU preset configured for NVIDIA A100."

# -------------------------------
#  Create build directory
# -------------------------------
mkdir -p build
cd build

# -------------------------------
#  Configure CMake for ML-IAP
# -------------------------------
print_status "Configuring LAMMPS with ML-IAP + Kokkos multi-GPU..."
cmake ../cmake \
  -C ../cmake/presets/basic.cmake \
  -C ../cmake/presets/mahti-gpu.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PROJAPPL/lammps-mliap \
  -DBUILD_MPI=ON \
  -DPKG_KOKKOS=ON \
  -DPKG_ML-IAP=ON \
  -DPKG_ML-SNAP=ON \
  -DPKG_PYTHON=ON \
  -DMLIAP_ENABLE_PYTHON=ON \
  -DBUILD_SHARED_LIBS=ON

print_success "CMake configuration complete."

# -------------------------------
#  Build & Install
# -------------------------------
print_status "Building LAMMPS (this may take several minutes)..."
make -j 8
print_success "LAMMPS built successfully."

print_status "Installing LAMMPS..."
make install
print_success "LAMMPS installed."

print_status "Installing LAMMPS Python package..."
make install-python
print_success "LAMMPS Python package installed."

# -------------------------------
#  Verify installation
# -------------------------------
if [ -f "$PROJAPPL/lammps-mliap/bin/lmp" ]; then
    print_success "LAMMPS executable found: $PROJAPPL/lammps-mliap/bin/lmp"
else
    print_error "LAMMPS executable not found — installation may have failed."
    exit 1
fi

# -------------------------------
#  Installation Summary
# -------------------------------
echo ""
echo "=============================================================="
print_success "LAMMPS-MLIAP (MACE + Multi-GPU) INSTALLED SUCCESSFULLY!"
echo "=============================================================="
echo ""
echo "Installation Summary:"
echo "  Username: $USERNAME"
echo "  Project: $PROJECT_NAME"
echo "  LAMMPS Path: $PROJAPPL/lammps-mliap"
echo "  Python venv: $PROJAPPL/venv"
echo "  Executable: $PROJAPPL/lammps-mliap/bin/lmp"
echo ""
echo "To use LAMMPS with ML-IAP:"
echo "  module purge"
echo "  module load gcc/11.2.0 openmpi/4.1.2 fftw/3.3.10-mpi cuda/11.5.0 cudnn/8.3.3.40-11.5 intel-oneapi-mkl/2021.4.0"
echo "  source $PROJAPPL/venv/bin/activate"
echo "  export PATH=$PROJAPPL/lammps-mliap/bin:\$PATH"
echo ""
echo "Example run command (multi-GPU):"
echo "  mpirun -np 2 lmp -k on g 2 -sf kk -pk kokkos newton on neigh half -in input.in"
echo ""
echo "Example LAMMPS input snippet:"
echo "  pair_style  mliap unified your_model-mliap_lammps.pt 0"
echo "  pair_coeff  * * C H O N"
echo ""
echo "=============================================================="
