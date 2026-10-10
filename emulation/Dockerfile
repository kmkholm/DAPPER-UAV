FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 iptables tcpdump wireguard-tools iperf3 procps curl ca-certificates iputils-ping ethtool \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pymavlink==2.4.49 numpy
RUN mkdir -p /app/bin /app/params /data \
 && curl -fsSL -o /app/bin/arducopter https://firmware.ardupilot.org/Copter/stable/SITL_x86_64_linux_gnu/arducopter \
 && curl -fsSL -o /app/bin/copter-git-version.txt https://firmware.ardupilot.org/Copter/stable/SITL_x86_64_linux_gnu/git-version.txt \
 && curl -fsSL -o /app/bin/arduplane https://firmware.ardupilot.org/Plane/stable/SITL_x86_64_linux_gnu/arduplane \
 && curl -fsSL -o /app/bin/plane-git-version.txt https://firmware.ardupilot.org/Plane/stable/SITL_x86_64_linux_gnu/git-version.txt \
 && chmod +x /app/bin/arducopter /app/bin/arduplane \
 && curl -fsSL -o /app/params/copter.parm https://raw.githubusercontent.com/ArduPilot/ardupilot/Copter-4.7.1/Tools/autotest/default_params/copter.parm \
 && curl -fsSL -o /app/params/quadplane.parm https://raw.githubusercontent.com/ArduPilot/ardupilot/Plane-4.7.1/Tools/autotest/default_params/quadplane.parm
WORKDIR /app
COPY node.py uplink.py gcs.py bg.py run.sh /app/
RUN sed -i 's/\r$//' /app/run.sh && chmod +x /app/run.sh
CMD ["/app/run.sh"]
