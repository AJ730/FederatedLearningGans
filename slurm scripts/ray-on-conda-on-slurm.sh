#!/bin/bash
# It is assumed that a conda environment is required to activate ray.
# The script has four command line options:
# --python_runfile="..." name of the python file to run
# --python_arguments="..." optional arguments to the python script
# --conda_env="..." name of the conda environment (default: base)
# --rundir="..." name of the working directory (default: current)
# --temp_dir="..." location to store ray temporary files (default: /tmp/ray)

# Author: Simon Tindemans, s.h.tindemans@tudelft.nl
# Version: 20 April 2022
#
# Ray-on-SLURM instruction and scripts used for inspiration:
# https://docs.ray.io/en/latest/cluster/slurm.html
# https://github.com/NERSC/slurm-ray-cluster
# https://github.com/pengzhenghao/use-ray-with-slurm

# set defaults
python_runfile="MISSING"
python_arguments=""
conda_env="base"
rundir="."
temp_dir="/tmp/ray"

# parse command line parameters
# use solution adapted from https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts 
for argument in "$@"
do
	  key=$(echo ${argument} | cut -f1 -d=)

	    key_length=${#key}
	      value="${argument:${key_length}+1}"

	        declare ${key#"--"}=${value}
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

	    node_1=${nodes_array[0]}
	    ip=$(srun --mpi=pmix --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

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

					      # launch head node, leaving one core unused for the main python script
					      echo "STARTING HEAD at $node_1"
					      srun --mpi=pmix --job-name="ray-head" --nodes=1 --ntasks=1 --cpus-per-task=$((SLURM_CPUS_PER_TASK-1)) -w "$node_1" \
						          conda-run.sh "$conda_env" \
							      "ray start --head --temp-dir="${temp_dir}" --include-dashboard=false --num-cpus=$((SLURM_CPUS_PER_TASK-1)) --node-ip-address=$ip --port=$port --redis-password=$redis_password --block"  &
					      sleep 10  # was sleep 30

					      # launch worker nodes
					      worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
					      for ((i = 1; i <= worker_num; i++)); do
						        node_i=${nodes_array[$i]}
							  echo "STARTING WORKER $i at $node_i"
							    srun --mpi=pmix --job-name="ray-worker" --nodes=1 --ntasks=1 -w "$node_i" \
								        conda-run.sh "$conda_env" \
									    "ray start --temp-dir="${temp_dir}" --num-cpus=$SLURM_CPUS_PER_TASK --address=$ip_head --redis-password=$redis_password --block" &
							      sleep 5
						      done

						      # export RAY_ADDRESS, so that ray can be initialised using ray.init(), without address
						      RAY_ADDRESS=$ip_head
						      export RAY_ADDRESS

						      # launch main program file on a single core. Wait for it to exit
						      srun --mpi=pmix --job-name="main" --unbuffered --nodes=1 --ntasks=1 --cpus-per-task=1 -w "$node_1" \
							          conda-run.sh "$conda_env" \
								      "ray status; cd ${rundir} ; python -u  ${python_runfile}" 

						      # if RAY_TIMELINE == 1, save the timeline
						      if [[ $RAY_TIMELINE == "1" ]]
						      then
							      srun --mpi=pmix --job-name="timeline" --nodes=1 --ntasks=1 --cpus-per-task=1 -w "$node_1" \
								          conda-run.sh "$conda_env" \
									      "ray timeline"
							      cp -n /tmp/ray-timeline-* ${temp_dir}
						      fi

						      # stop the ray cluster
						      srun --mpi=pmix --job-name="ray-stop" --nodes=1 --ntasks=1 --cpus-per-task=1 -w "$node_1" \
							          conda-run.sh "$conda_env" \
								      "ray stop"

						      # wait for everything so we don't cancel head/worker jobs that have not had time to clean up
						      wait
						      
