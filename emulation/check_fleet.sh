P=f4288d46-7b91-4ed9-ac16-9074ad3dff9d
echo "containers: $(docker ps -q | wc -l)"; free -m | head -2; echo "load: $(cat /proc/loadavg)"
up=0; tun=0; sitl=0
for d in A1-d1 A1-d2 A1-d3 A2-d1 A2-d2 A2-d3 A3-d1 A3-d2 A3-d3 A4-d1 A4-d2 A4-d3 B1-d1 B1-d2 B1-d3 B2-d1 B2-d2 B2-d3 B3-d1 B3-d2 B3-d3 B4-d1 B4-d2 B4-d3; do
  o=$(docker exec GNS3.$d.$P sh -c 'grep -c "tunnel up" /data/node.log 2>/dev/null; grep -c "connected to SITL" /data/uplink.log 2>/dev/null' 2>/dev/null | tr "\n" " ")
  echo "$d: $o"
done
