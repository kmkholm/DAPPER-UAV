#!/bin/sh
# GNS3 start command. CFG holds the base64 JSON node configuration written by topology.py.
# The node's logs are streamed to stdout so that the GNS3 telnet console shows them live.
mkdir -p /data
echo "$CFG" | base64 -d > /tmp/cfg.json
python3 -u /app/node.py /tmp/cfg.json >> /data/node.log 2>&1 &
sleep 2
touch /data/node.log
exec tail -n 20 -F /data/node.log /data/sitl.log /data/uplink.log /data/gcs.log /data/gcs_events.csv /data/tcpdump_*.log 2>/dev/null
