FROM linuxserver/letsencrypt

RUN ls -l .

COPY ./config/dns-conf/cloudflare.ini /config/dns-conf/cloudflare.ini
RUN cat /config/dns-conf/cloudflare.ini
RUN ls -l /config/dns-conf/