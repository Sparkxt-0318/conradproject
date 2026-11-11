# Sample field

Stitched outputs for trying the layer and report steps without running OpenDroneMap:

```bash
soilsight scout run data/sample_field --skip-stitch
```

`odm/` holds the orthophoto and surface model that OpenDroneMap 3.6.2 produced from the 18-photo Brighton Beach test set, reduced here from 2 cm and 5 cm pixels to 25 cm to keep the repository small, plus OpenDroneMap's `stats.json` unchanged. `annotations.geojson` is an invented boundary, inlet, outlet and ditch drawn on that orthophoto to exercise every layer; the scene is a park, not a farm field.

The photos are by Piero Toffanin, from https://github.com/pierotofy/drone_dataset_brighton_beach, under the BSD 2-Clause licence:

```
BSD 2-Clause License

Copyright (c) 2017, Piero Toffanin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
