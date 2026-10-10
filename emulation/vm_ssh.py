"""SSH/SFTP helper for the GNS3 VM (stock credentials gns3/gns3).

  python vm_ssh.py run "docker version"
  python vm_ssh.py put <local> <remote>
  python vm_ssh.py get <remote> <local>
IP is read from out/gns3vm_ip.txt unless GNS3VM_IP is set.
"""
import os, sys, time
import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
IPF = os.path.join(HERE, "gns3vm_ip.txt")
IP = os.environ.get("GNS3VM_IP") or open(IPF).read().strip()
USER, PW = os.environ.get("GNS3VM_USER", "gns3"), os.environ.get("GNS3VM_PW", "gns3")


def connect(tries=8):
    last = None
    for i in range(tries):
        try:
            c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(IP, username=USER, password=PW, timeout=20, banner_timeout=40, auth_timeout=40,
                      allow_agent=False, look_for_keys=False)
            return c
        except Exception as e:                       # sshd MaxStartups / transient banner errors
            last = e; time.sleep(3 + 2 * i)
    raise last


def run(cmd, timeout=3600):
    c = connect()
    _, out, err = c.exec_command(cmd, timeout=timeout, get_pty=False)
    o, e = out.read().decode(errors="replace"), err.read().decode(errors="replace")
    rc = out.channel.recv_exit_status(); c.close()
    return rc, o, e


def put(local, remote):
    c = connect(); s = c.open_sftp()
    t0 = time.time(); s.put(local, remote); s.close(); c.close()
    print("uploaded %s -> %s (%.0f MB in %.0fs)" % (local, remote, os.path.getsize(local) / 1048576, time.time() - t0))


def get(remote, local):
    c = connect(); s = c.open_sftp(); s.get(remote, local); s.close(); c.close()
    print("downloaded", remote, "->", local)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        rc, o, e = run(a[1]); print(o, end=""); print(e, end="", file=sys.stderr); sys.exit(rc)
    elif a[0] == "put":
        put(a[1], a[2])
    elif a[0] == "get":
        get(a[1], a[2])
