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
JOB_NAME="ai_maid_theo_$now"
PACKAGE_STAGING_PATH="gs://theos_jobs"
```
2. Run job locally
```
$ gcloud ml-engine local train --package-path trainer \
                               --module-name trainer.task \
                               -- \

OR ::::

$ cd mlengine_theo/trainer

$ python3 task.py

```
or distributed
```
inside image-painter-gans

remember to change paths in task... (need to fix this)

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

5. To launch tensorboard for example run:

```
tensorboard --logdir='gs://theos_jobs/testing/tb_logs/' --port=8088
```
then open localhost:8088/