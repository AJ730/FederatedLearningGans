#!/bin/bash
# This script is meant to be called from a SLURM job file.
# It launches a ray server on each compute node and starts a python script on the head node.
# It is assumed that a conda environment is required to activate ray.

# The script has four command line options:
# --python_runfile="..." name of the python file to run
# --python_arguments="..." optional arguments to the python script
# --conda_env="..." name of the conda environment (default: base)
# --run_dir="..." name of the working directory (default: current)
# --temp_dir="..." location to store ray temporary files (default: /tmp/ray)

# Author: Simon Tindemans, s.h.tindemans@tudelft.nl
# Version: 18 May 2022
#
# Ray-on-SLURM instruction and scripts used for inspiration:
# https://docs.ray.io/en/latest/cluster/slurm.html
# https://github.com/NERSC/slurm-ray-cluster
# https://github.com/pengzhenghao/use-ray-with-slurm

#TODO: allow multiple ray servers on a node: https://github.com/ray-project/ray/issues/10154
# use the following to count open ports: ss -at '( sport = :2244 )' | $(($(wc -l) - 1))

# set defaults
python_runfile="MISSING"
python_arguments=""
conda_env="base"
run_dir="."
temp_dir="/tmp/ray"

# parse command line parameters
# use solution adapted from https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts 
for argument in "$@"
do
  key=$(echo ${argument} | cut -f1 -d=)

  key_length=${#key}
  value="${argument:${key_length}+1}"

  declare ${key#"--"}="${value}"
done

# Abort if no main python file name is given
if [[ ${python_runfile} == "MISSING" ]]
then
  echo "Missing python_runfile option. Aborting."
  exit
fi

if [[ $RAY_TIMELINE == "1" ]]
then
echo "RAY PROFILING MODE ENABLED"
fi

# generate password for node-to-node communication
redis_password=$(uuidgen)
export redis_password

# get node names and identify IP addresses
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# assign head and worker nodes
head_node=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address) # making redis-address
worker_nodes=$(IFS=,; echo "${nodes_array[*]:1}")

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

# set up head node
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"
# export RAY_ADDRESS, so that ray can be initialised using ray.init(), without address
RAY_ADDRESS=$ip_head
export RAY_ADDRESS

head_worker_threads=$((SLURM_CPUS_PER_TASK-1))

# set up all commands that will be run on the head node
HEAD_CMD=$(cat << EOM
conda activate ${conda_env}
ray start --head --temp-dir="${temp_dir}" --include-dashboard=false --num-cpus=${head_worker_threads} --node-ip-address=$ip --port=$port --redis-password=$redis_password
sleep 600
ray status
cd ${run_dir}
python -u  ${python_runfile} ${python_arguments}
if [[ $RAY_TIMELINE == "1" ]]
then
ray timeline
cp -n /tmp/ray-timeline-* ${temp_dir}
fi
ray stop
EOM
)

# set up all commands that will be run on each worker node
WORKER_CMD=$(cat << EOM
conda activate ${conda_env}
ray start --temp-dir="${temp_dir}" --num-cpus=$SLURM_CPUS_PER_TASK --address=$ip_head --redis-password=$redis_password --block
EOM
)

# TODO: implement barrier:
# https://github.com/EricCrosson/bash-barrier
# on all nodes, launch local barrier that wait for sign from head node
# on head node, start ray head and release all barriers
# on head node, launch local barrier 
# on other nodes, start ray workers and release central barrier
# on head, run job
# on workers, use sleep infinity

# launch head node, leaving one core unused for the main python script
echo "STARTING HEAD at $head_node"
srun --job-name="ray-head" --unbuffered --nodes=1 --ntasks=1 -w "$head_node" \
	conda-run.sh "${HEAD_CMD}" &

# if we are running on more than one node, start worker nodes
if [[ $SLURM_JOB_NUM_NODES != "1" ]]
then
 sleep 60  # wait for the head node to fully start before launching worker nodes
 worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
 echo "STARTING ${worker_num} WORKER NODES"
 srun --job-name="ray-workers" --nodes=${worker_num} --ntasks=${worker_num} -w "${worker_nodes}" \
	conda-run.sh "${WORKER_CMD}" &
fi

# wait for everything so we don't cancel head/worker jobs that have not had time to clean up
wait