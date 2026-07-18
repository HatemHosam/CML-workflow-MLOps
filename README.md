# CML-workflow-MLOps
A general Template of CML (Continuous Machine Learning) for automatic training, deployment, and monitoring. 

#### Configuring Cloud Storage Providers

There are many
[supported could storage providers](https://dvc.org/doc/command-reference/remote/modify#available-parameters-per-storage-type).
Here are a few examples for some of the most frequently used providers:

<details>
  <summary>
  S3 and S3-compatible storage (Minio, DigitalOcean Spaces, IBM Cloud Object Storage...)
  </summary>

```yaml
# Github
env:
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  AWS_SESSION_TOKEN: ${{ secrets.AWS_SESSION_TOKEN }}
```

> :point_right: `AWS_SESSION_TOKEN` is optional.

> :point_right: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` can also be used
> by `cml runner` to launch EC2 instances. See [Environment Variables].

</details>

<details>
  <summary>
  Azure
  </summary>

```yaml
env:
  AZURE_STORAGE_CONNECTION_STRING:
    ${{ secrets.AZURE_STORAGE_CONNECTION_STRING }}
  AZURE_STORAGE_CONTAINER_NAME: ${{ secrets.AZURE_STORAGE_CONTAINER_NAME }}
```

</details>

<details>
  <summary>
  Aliyun
  </summary>

```yaml
env:
  OSS_BUCKET: ${{ secrets.OSS_BUCKET }}
  OSS_ACCESS_KEY_ID: ${{ secrets.OSS_ACCESS_KEY_ID }}
  OSS_ACCESS_KEY_SECRET: ${{ secrets.OSS_ACCESS_KEY_SECRET }}
  OSS_ENDPOINT: ${{ secrets.OSS_ENDPOINT }}
```

</details>

<details>
  <summary>
  Google Storage
  </summary>

> :warning: Normally, `GOOGLE_APPLICATION_CREDENTIALS` is the **path** of the
> `json` file containing the credentials. However in the action this secret
> variable is the **contents** of the file. Copy the `json` contents and add it
> as a secret.

```yaml
env:
  GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GOOGLE_APPLICATION_CREDENTIALS }}
```

</details>

<details>
  <summary>
  Google Drive
  </summary>

> :warning: After configuring your
> [Google Drive credentials](https://dvc.org/doc/command-reference/remote/add)
> you will find a `json` file at
> `your_project_path/.dvc/tmp/gdrive-user-credentials.json`. Copy its contents
> and add it as a secret variable.

```yaml
env:
  GDRIVE_CREDENTIALS_DATA: ${{ secrets.GDRIVE_CREDENTIALS_DATA }}
```

</details>

## Advanced Setup

### Self-hosted (On-premise or Cloud) Runners

GitHub Actions are run on GitHub-hosted runners by default. However, there are
many great reasons to use your own runners: to take advantage of GPUs,
orchestrate your team's shared computing resources, or train in the cloud.

> :point_up: **Tip!** Check out the
> [official GitHub documentation](https://help.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners)
> to get started setting up your own self-hosted runner.

#### Allocating Cloud Compute Resources with CML

When a workflow requires computational resources (such as GPUs), CML can
automatically allocate cloud instances using `cml runner`. You can spin up
instances on AWS, Azure, GCP, or Kubernetes.

For example, the following workflow deploys a `g4dn.xlarge` instance on AWS EC2
and trains a model on the instance. After the job runs, the instance
automatically shuts down.

You might notice that this workflow is quite similar to the
[basic use case](#usage) above. The only addition is `cml runner` and a few
environment variables for passing your cloud service credentials to the
workflow.

Note that `cml runner` will also automatically restart your jobs (whether from a
[GitHub Actions 35-day workflow timeout](https://docs.github.com/en/actions/reference/usage-limits-billing-and-administration#usage-limits)
or a
[AWS EC2 spot instance interruption](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html)).

```yaml
name: Train-in-the-cloud
on: [push]
jobs:
  deploy-runner:
    runs-on: ubuntu-latest
    steps:
      - uses: iterative/setup-cml@v1
      - uses: actions/checkout@v3
      - name: Deploy runner on EC2
        env:
          REPO_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          cml runner launch \
            --cloud=aws \
            --cloud-region=us-west \
            --cloud-type=g4dn.xlarge \
            --labels=cml-gpu
  train-model:
    needs: deploy-runner
    runs-on: [self-hosted, cml-gpu]
    timeout-minutes: 50400 # 35 days
    container:
      image: ghcr.io/iterative/cml:0-dvc2-base1-gpu
      options: --gpus all
    steps:
      - uses: actions/checkout@v3
      - name: Train model
        env:
          REPO_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }}
        run: |
          pip install -r requirements.txt
          python train.py

          cat metrics.txt > report.md
          cml comment create report.md
```

In the workflow above, the `deploy-runner` step launches an EC2 `g4dn.xlarge`
instance in the `us-west` region. The `model-training` step then runs on the
newly-launched instance. See [Environment Variables] below for details on the
`secrets` required.

> :tada: **Note that jobs can use any Docker container!** To use functions such
> as `cml send-comment` from a job, the only requirement is to
> [have CML installed](#local-package).

#### Docker Images

The CML Docker image (`ghcr.io/iterative/cml` or `iterativeai/cml`) comes loaded
with Python, CUDA, `git`, `node` and other essentials for full-stack data
science. Different versions of these essentials are available from different
image tags. The tag convention is `{CML_VER}-dvc{DVC_VER}-base{BASE_VER}{-gpu}`:

| `{BASE_VER}` | Software included (`-gpu`)                    |
| ------------ | --------------------------------------------- |
| 0            | Ubuntu 18.04, Python 2.7 (CUDA 10.1, CuDNN 7) |
| 1            | Ubuntu 20.04, Python 3.8 (CUDA 11.2, CuDNN 8) |

For example, `iterativeai/cml:0-dvc2-base1-gpu`, or
`ghcr.io/iterative/cml:0-dvc2-base1`.

#### Arguments

The `cml runner launch` function accepts the following arguments:

```
  --labels                                  One or more user-defined labels for
                                            this runner (delimited with commas)
                                                       [string] [default: "cml"]
  --idle-timeout                            Time to wait for jobs before
                                            shutting down (e.g. "5min"). Use
                                            "never" to disable
                                                 [string] [default: "5 minutes"]
  --name                                    Name displayed in the repository
                                            once registered
                                                    [string] [default: cml-{ID}]
  --no-retry                                Do not restart workflow terminated
                                            due to instance disposal or GitHub
                                            Actions timeout            [boolean]
  --single                                  Exit after running a single job
                                                                       [boolean]
  --reuse                                   Don't launch a new runner if an
                                            existing one has the same name or
                                            overlapping labels         [boolean]
  --reuse-idle                              Creates a new runner only if the
                                            matching labels don't exist or are
                                            already busy               [boolean]
  --docker-volumes                          Docker volumes, only supported in
                                            GitLab         [array] [default: []]
  --cloud                                   Cloud to deploy the runner
                         [string] [choices: "aws", "azure", "gcp", "kubernetes"]
  --cloud-region                            Region where the instance is
                                            deployed. Choices: [us-east,
                                            us-west, eu-west, eu-north]. Also
                                            accepts native cloud regions
                                                   [string] [default: "us-west"]
  --cloud-type                              Instance type. Choices: [m, l, xl].
                                            Also supports native types like i.e.
                                            t2.micro                    [string]
  --cloud-permission-set                    Specifies the instance profile in
                                            AWS or instance service account in
                                            GCP           [string] [default: ""]
  --cloud-metadata                          Key Value pairs to associate
                                            cml-runner instance on the provider
                                            i.e. tags/labels "key=value"
                                                           [array] [default: []]
  --cloud-gpu                               GPU type. Choices: k80, v100, or
                                            native types e.g. nvidia-tesla-t4
                                                                        [string]
  --cloud-hdd-size                          HDD size in GB              [number]
  --cloud-ssh-private                       Custom private RSA SSH key. If not
                                            provided an automatically generated
                                            throwaway key will be used  [string]
  --cloud-spot                              Request a spot instance    [boolean]
  --cloud-spot-price                        Maximum spot instance bidding price
                                            in USD. Defaults to the current spot
                                            bidding price [number] [default: -1]
  --cloud-startup-script                    Run the provided Base64-encoded
                                            Linux shell script during the
                                            instance initialization     [string]
  --cloud-aws-security-group                Specifies the security group in AWS
                                                          [string] [default: ""]
  --cloud-aws-subnet,                       Specifies the subnet to use within
  --cloud-aws-subnet-id                     AWS           [string] [default: ""]

```

#### Environment Variables

> :warning: You will need to
> [create a personal access token (PAT)](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line)
> with repository read/write access and workflow privileges. In the example
> workflow, this token is stored as `PERSONAL_ACCESS_TOKEN`.

:information_source: If using the `--cloud` option, you will also need to
provide access credentials of your cloud compute resources as secrets. In the
above example, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (with privileges
to create & destroy EC2 instances) are required.

For AWS, the same credentials can also be used for
[configuring cloud storage](#configuring-cloud-storage-providers).

