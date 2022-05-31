#!/bin/bash
# $(ClusterId) $(ProcId) $(config_file)
export PYTHONUSERBASE="/afs/cern.ch/work/l/lgolino/public/vipm"
export PATH="$PYTHONUSERBASE/bin/:$PATH"
export EOS_MGM_URL=root://eosproject.cern.ch

# ------------------------------------------------------------------------------
echo "Starting job $1.$2 on $(date)..."

# Activate environment
source /afs/cern.ch/work/l/lgolino/public/vipm/bin/activate

# Make output directory
output_directory="/eos/user/l/lgolino/Documents/BIGroupAllWork/VIPMSims/INSERTHERE/files/results"
#eos mkdir -p $output_directory
# get the name of the result file
result_path=$(basename $3 .xml)
result_path="$result_path.csv"

# ------------------------------------------------------------------------------
# Run
virtual-ipm $3 --console-log-level warning
# Compress and copy the result to the output directory
python /eos/user/l/lgolino/Documents/BIGroupAllWork/Python/TestDataExtraction/DataExtractionV1.py "$result_path"
gzip "$result_path.pkl"
xrdcp "$result_path.pkl.gz" "root://eosproject.cern.ch/$output_directory/"
rm "$result_path.pkl.gz"
rm "$result_path"


# ------------------------------------------------------------------------------
echo "Finished job $1.$2 on $(date)..."
