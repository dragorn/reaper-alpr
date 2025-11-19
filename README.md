# Reaper - Fast-ALPR
based on fast-alpr  https://github.com/ankandrew/fast-alpr.git

```
git submodule update --init --recrusive
```

edit `config/reaperalpr.yaml`

point the `host` at the IP of your camera, and set the protocol.  it appears
that older (ie, cheaper) cameras use what we're going to call `protocol 1`, and
newer raspberry-pi powered cameras use what we're going to call `protocol 2`.

then fire up the docker image:

```
docker compose up
```

## early code

this is highly early experimental code, expect some hiccups and significant lack
of polish

## mjpeg stream

if you've built the docker before the mjpeg was added, you'll need to rebuild it
to pull in the new dependencies.

the mjpeg server will be served on port 8080 under `/reaper`, with docker
forwarding, [http://localhost:8080/reaper](http://localhost:8080/reaper)

if your camera supports a full image (protocol v2, apparently), it will be at
[http://localhost:8080/reaper-color](http://localhost:8080/reaper-color)

## known to-do stuff

* Instructions on how to pass GPU into docker for accelerated support
* storing matched images by plate
* logging & export

## known issues

* ~~sometimes the images desync and switch to color~~ fixed with new stream
  parser
