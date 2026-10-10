P=f4288d46-7b91-4ed9-ac16-9074ad3dff9d
LOG="$(dirname "$0")/../results/collect.log"
while true; do
  st=$(python vm_ssh.py run "echo C=\$(docker ps -q|wc -l); free -m|awk '/Mem:/{print \"MEM=\"\$3\"/\"\$2}'; echo L=\$(cut -d' ' -f1 /proc/loadavg); for g in gcs-A gcs-B; do docker exec GNS3.\$g.\$P sh -c 'grep -c mission_start /data/gcs_events.csv 2>/dev/null'; done | paste -sd+ | sed 's/^/MISS=/'" 2>/dev/null | tr '\n' ' ')
  echo "$(date +%H:%M) $st" >> "$LOG"
  c=$(echo "$st" | grep -oE 'C=[0-9]+' | cut -d= -f2)
  if [ -n "$c" ] && [ "$c" -lt 40 ]; then echo "$(date +%H:%M) ALARM containers=$c" >> "$LOG"; fi
  sleep 300
done
