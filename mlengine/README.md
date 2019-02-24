# Official ML Engine Version

### Train Locally
```
$ JOB_DIR='jobs'
$ python -m trainer.task
```

### Train on ML Engine
1. Set job name
```
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="ai_maid_test_$now"
PACKAGE_STAGING_PATH="gs://lsun-roomsets"
```
2. Run job locally
```
gcloud ml-engine local train --package-path trainer \
                             --module-name trainer.task \
                             -- \

```
or distributed
```
$ gcloud ml-engine jobs submit training $JOB_NAME \
                                    --module-name trainer.task \
                                    --package-path trainer \
                                    --python-version 3.5 \
                                    --runtime-version 1.4 \
                                    --region us-central1 \
                                    --staging-bucket $PACKAGE_STAGING_PATH \
                                    -- \

```
3. View job status or stream logs
```
<!-- View Status -->
$ gcloud ml-engine jobs describe $JOB_NAME

<!-- Stream Logs -->
$ gcloud ml-engine jobs stream-logs $JOB_NAME
```

4. Or cancel a job
```
$ gcloud ml-engine jobs cancel $JOB_NAME
```
