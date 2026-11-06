#!/bin/bash
# Author: Vito Foderà
# Created: 2026-11-04
#
# Jobscript of tests/hpc/qe_check.py: resources and environment only. The
# worker line (or, in the case single, the pw.x line) is added to a copy of
# it, and the walltime of each case goes in the --time line. Example for a Slurm
# machine with 4 GPUs per node, QE from a module and mpirun as launcher: fill
# in account and venv, check partition, QoS and module on your machine.
#SBATCH --job-name=qe_check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time={time}
#SBATCH --account=<account>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
# Round 5 of docs/hpc_check.md requeues a job: where Slurm does not requeue by
# default (JobRequeue = 0 in "scontrol show config"), the jobscript of that run
# needs "#SBATCH --requeue".
# %j: each job of a group writes its own files.
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close

module purge
module load profile/chem-phys
module load quantum-espresso/7.2--openmpi--4.1.4--nvhpc--23.1-openblas-cuda-11.8

source <venv>/bin/activate
