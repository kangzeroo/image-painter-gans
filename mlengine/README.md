# Official ML Engine Version

### Train on ML Engine
1. Set job name
```
$ JOB_NAME='ai_maid_test_2'
$ JOB_DIR='gs://lsun-roomsets/jobs/'
```
2. Run job
```
$ gcloud ml-engine jobs submit training $JOB_NAME \
                                    --module-name trainer.task \
                                    --package-path trainer \
                                    --job-dir $JOB_DIR \
                                    --python-version 3.5 \
                                    --runtime-version 1.4 \
                                    --region us-central1 \


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
