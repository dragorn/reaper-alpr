# Reaper - Fast-ALPR
based on fast-alpr  https://github.com/ankandrew/fast-alpr.git

```
git submodule update --init --recrusive
docker build -t reaperml .
REAPER=reaper-hostname docker run -v $(pwd):/data -E REAPER -it reaperml
```
