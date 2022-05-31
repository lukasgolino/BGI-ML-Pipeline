The using VIPM pdf file (written by Swann) contains all required information on running VIPM sims with parameter sweeps with 2 major differences:

1. We now use the DataExtractionV1.py to extract the important data during the condor job. This can be tailored to suit your needs but currently only takes initial and final x positions, and bins them for us. Reducing disk space by 99+%. Make not of where you store this Python script tpo ensure the condor node can access it (DFS is no longer accessable by condor, only AFS. Hence the next change). Notice line 24 in run_simulation.py:

`python /eos/user/l/lgolino/Documents/BIGroupAllWork/Python/TestDataExtraction/DataExtractionV1.py "$result_path"`

2. Condor can no longer read files from DFS. The config files must either be copied to AFS (which is still accessable) or placed somewhere accessable. Notice lines 14-16 in job.sub:

`# transfer_input_files    = /eos/user/l/lgolino/Documents/BIGroupAllWork/VIPMSims/INSERTHERE/files/configurations/$(config_file)`

`# RUN THIS: scp -r /eos/user/l/lgolino/Documents/BIGroupAllWork/VIPMSims/INSERTHERE/files/configurations/ ./configurations`

`transfer_input_files    = /afs/cern.ch/work/l/lgolino/private/BI/simulations/condor_jobs/condor_job_INSERTHERE/configurations/$(config_file)`