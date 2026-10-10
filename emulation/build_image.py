"""Upload the Docker build context to the GNS3 VM and build dapper-node:latest there."""
import os, sys, tarfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vm_ssh
FILES = ["Dockerfile", "node.py", "uplink.py", "gcs.py", "bg.py", "run.sh"]
tarp = os.path.join(HERE, "ctx.tar")
with tarfile.open(tarp, "w") as t:
    for f in FILES:
        t.add(os.path.join(HERE, f), arcname=f)
vm_ssh.put(tarp, "dapper_ctx.tar")
rc, o, e = vm_ssh.run("rm -rf ~/dapper_ctx && mkdir ~/dapper_ctx && tar -xf ~/dapper_ctx.tar -C ~/dapper_ctx && cd ~/dapper_ctx && "
                      "docker build -t dapper-node:latest . 2>&1 | tail -15 && docker run --rm dapper-node:latest sh -c "
                      "'cat /app/bin/*git-version.txt | grep -E \"^commit|APMVERSION\"; python3 -c \"import pymavlink;print(pymavlink.__version__)\"; ls -la /app/bin /app/params'",
                      timeout=3600)
print(o); print(e, file=sys.stderr)
sys.exit(rc)
