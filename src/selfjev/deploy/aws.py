"""One-command AWS deployment of a selfjev server (needs `pip install selfjev[deploy]` and AWS credentials).

  selfjev deploy aws up --name prod --instance g6.xlarge --region us-east-2     # prints the endpoint and the API key
  selfjev deploy aws status --name prod
  selfjev deploy aws down --name prod                                          # terminates; deletes the SG and key pair

The box: NVIDIA's Deep Learning Base AMI (drivers, CUDA), this repository at a pinned ref with only the
selfjev-4b weights from Git LFS, `uv sync --extra serve --extra gpu`, the base model pre-downloaded, and
`selfjev serve` as a systemd service on port 8000 behind a Bearer key. Setup takes 10 to 15 minutes. State lives in
~/.selfjev/deployments/<name>.json; every AWS resource is tagged Project=selfjev, Name=selfjev-<name>.
"""

import json
import secrets
import time
import urllib.request
from pathlib import Path

REPO = "https://github.com/Jwuthri/SelfJev.git"
BASE_MODEL = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
AMI_PARAMETER = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
STATE = Path.home() / ".selfjev" / "deployments"
PORT = 8000
# on-demand $/h in us-east-2, 2026-09; other regions differ. The model needs >= 16 GB of GPU memory.
MACHINES = {
    "g6.xlarge": ("1x L4, 24 GB", 0.805, "the default: cheapest box that serves selfjev-4b well"),
    "g5.xlarge": ("1x A10G, 24 GB", 1.006, "when L4 capacity is short"),
    "g6e.xlarge": ("1x L40S, 48 GB", 1.861, "faster, long texts, more concurrent traffic"),
    "g6e.2xlarge": ("1x L40S, 48 GB, 8 vCPU", 2.242, "as g6e.xlarge with more CPU"),
    "p5.4xlarge": ("1x H100, 80 GB", 6.88, "lowest latency; often out of capacity"),
}


def user_data(api_key: str, ref: str, max_hours: float | None = None) -> str:
    """The first-boot script: install, fetch the weights and the base model, start the service."""
    cap = f"shutdown -h +{int(max_hours * 60)}  # cost cap: the instance terminates on shutdown\n" if max_hours else ""
    return f"""#!/bin/bash
set -euxo pipefail
exec > /var/log/selfjev-setup.log 2>&1
{cap}apt-get update -q && apt-get install -yq git-lfs
cat > /tmp/selfjev-setup.sh <<'SETUP'
set -euxo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none {REPO} ~/selfjev
cd ~/selfjev && git checkout {ref}
git lfs install --local && git lfs pull --include 'weights/selfjev_4b/*'
~/.local/bin/uv sync --frozen --no-dev --extra serve --extra gpu
~/.local/bin/uv run --no-sync python -c "from huggingface_hub import snapshot_download; \\
snapshot_download('{BASE_MODEL[0]}', revision='{BASE_MODEL[1]}')"
SETUP
sudo -iu ubuntu bash /tmp/selfjev-setup.sh
cat > /etc/systemd/system/selfjev.service <<'UNIT'
[Unit]
Description=selfjev decisions API
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/selfjev
Environment=SELFJEV_API_KEYS={api_key}
ExecStart=/home/ubuntu/.local/bin/uv run --no-sync selfjev serve --host 0.0.0.0 --port {PORT}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now selfjev
"""


def _clients(region):
    import boto3

    session = boto3.Session(region_name=region)
    return session.client("ec2"), session.client("ssm")


def _tags(name, kind):
    return [{"ResourceType": kind, "Tags": [{"Key": "Project", "Value": "selfjev"}, {"Key": "Name", "Value": f"selfjev-{name}"}]}]


def _state(name) -> Path:
    return STATE / f"{name}.json"


def _healthy(url: str, key: str) -> bool:
    try:
        req = urllib.request.Request(f"{url}/health", headers={"Authorization": f"Bearer {key}"})
        return json.loads(urllib.request.urlopen(req, timeout=5).read()).get("status") == "ok"
    except Exception:
        return False


def up(
    name, instance="g6.xlarge", region="us-east-2", allow_cidr="0.0.0.0/0", ref="master", api_key=None, max_hours=None, ssh=False, wait=True
):
    """Launch a box; returns the deployment record (endpoint, key, ids). ssh=True adds a key pair and port 22 from this
    machine's IP (for reading /var/log/selfjev-setup.log). Raises if `name` is already deployed."""
    if _state(name).exists():
        raise SystemExit(f"deployment '{name}' exists ({_state(name)}); `selfjev deploy aws down --name {name}` first")
    ec2, ssm = _clients(region)
    api_key = api_key or "sj-" + secrets.token_urlsafe(24)
    ami = ssm.get_parameter(Name=AMI_PARAMETER)["Parameter"]["Value"]
    vpc = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    sg = ec2.create_security_group(
        GroupName=f"selfjev-{name}", Description=f"selfjev {name}: API port", VpcId=vpc, TagSpecifications=_tags(name, "security-group")
    )["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg, IpPermissions=[{"IpProtocol": "tcp", "FromPort": PORT, "ToPort": PORT, "IpRanges": [{"CidrIp": allow_cidr}]}]
    )
    record = {"name": name, "region": region, "instance_type": instance, "security_group": sg, "api_key": api_key, "ref": ref}
    STATE.mkdir(parents=True, exist_ok=True)
    extra = {}
    if ssh:
        my_ip = urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()
        ec2.authorize_security_group_ingress(
            GroupId=sg, IpPermissions=[{"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": f"{my_ip}/32"}]}]
        )
        pem = STATE / f"{name}-{region}.pem"
        pem.write_text(
            ec2.create_key_pair(KeyName=f"selfjev-{name}", KeyType="ed25519", TagSpecifications=_tags(name, "key-pair"))["KeyMaterial"]
        )
        pem.chmod(0o600)
        record |= {"key_pair": f"selfjev-{name}", "ssh": f"ssh -i {pem} ubuntu@<ip>"}
        extra["KeyName"] = f"selfjev-{name}"
    try:
        r = ec2.run_instances(
            ImageId=ami,
            InstanceType=instance,
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=[sg],
            UserData=user_data(api_key, ref, max_hours),
            InstanceInitiatedShutdownBehavior="terminate",
            BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True}}],
            TagSpecifications=_tags(name, "instance") + _tags(name, "volume"),
            **extra,
        )
    except Exception:
        ec2.delete_security_group(GroupId=sg)
        raise
    record["instance_id"] = r["Instances"][0]["InstanceId"]
    record["launched"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ec2.get_waiter("instance_running").wait(InstanceIds=[record["instance_id"]])
    ip = ec2.describe_instances(InstanceIds=[record["instance_id"]])["Reservations"][0]["Instances"][0]["PublicIpAddress"]
    record["endpoint"] = f"http://{ip}:{PORT}"
    if ssh:
        record["ssh"] = record["ssh"].replace("<ip>", ip)
    _state(name).write_text(json.dumps(record, indent=2))
    _state(name).chmod(0o600)  # holds the API key
    if wait:
        deadline = time.time() + 30 * 60
        while not _healthy(record["endpoint"], api_key) and time.time() < deadline:
            time.sleep(20)
        record["ready"] = _healthy(record["endpoint"], api_key)
    return record


def down(name):
    rec = json.loads(_state(name).read_text())
    ec2, _ = _clients(rec["region"])
    ec2.terminate_instances(InstanceIds=[rec["instance_id"]])
    ec2.get_waiter("instance_terminated").wait(InstanceIds=[rec["instance_id"]])
    ec2.delete_security_group(GroupId=rec["security_group"])
    if rec.get("key_pair"):
        ec2.delete_key_pair(KeyName=rec["key_pair"])
        (STATE / f"{name}-{rec['region']}.pem").unlink(missing_ok=True)
    _state(name).unlink()
    return rec


def status(name):
    rec = json.loads(_state(name).read_text())
    ec2, _ = _clients(rec["region"])
    inst = ec2.describe_instances(InstanceIds=[rec["instance_id"]])["Reservations"][0]["Instances"][0]
    hours = (time.time() - inst["LaunchTime"].timestamp()) / 3600
    price = MACHINES.get(rec["instance_type"], ("", None))[1]
    return rec | {
        "state": inst["State"]["Name"],
        "healthy": _healthy(rec["endpoint"], rec["api_key"]),
        "uptime_h": round(hours, 2),
        "cost_so_far_usd": round(hours * price, 2) if price else None,
    }


def listing():
    return [json.loads(p.read_text()) for p in sorted(STATE.glob("*.json"))] if STATE.exists() else []
