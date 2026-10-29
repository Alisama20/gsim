Generic reproducer for a Palace wave-port issue: two coupled microstrips (generic gdsfactory PDK stack) in a closed PEC box,
200 um line, two wave-port modes per face, metal as a Conductivity boundary (sigma 3.77e7 S/m).
- 3d_lossy_L200_m{1,2}: control, metal listed in Boundaries.WavePortPEC (port modes lossless). m1/m2 = which mode is Active.
- 3d_lossyport_L200_m{1,2}: same config with WavePortPEC removed (port modes see the metal loss).
- 2d_lossy_f{10,40}: BoundaryMode solves of the same cross-section with the lossy walls (reference modes).
Each folder: config.json, palace.msh, palace.log (Palace e1bdf9b7).
