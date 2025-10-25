# Reaper - Fast-ALPR
based on fast-alpr  https://github.com/ankandrew/fast-alpr.git

```
git submodule update --init --recrusive
docker build -t reaperml .
REAPER=reaper-hostname docker run -v $(pwd):/data -p 8080:8080 -E REAPER -it reaperml
```

## early code

this is highly early experimental code, expect some hiccups and significant lack
of polish

## mjpeg stream

if you've built the docker before the mjpeg was added, you'll need to rebuild it
to pull in the new dependencies.

the mjpeg server will be served on port 8080 under `/reaper`, with docker
forwarding, [http://localhost:8080/reaper](http://localhost:8080/reaper)
