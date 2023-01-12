@echo off
set CDIR="%~dp0"
docker run -v %CDIR%:/tmp sd --prompt "Emilia Clake drinking a coffee" --output /tmp/result.png
