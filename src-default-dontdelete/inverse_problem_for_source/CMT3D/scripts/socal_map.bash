gmtset PAPER_MEDIA letter BASEMAP_TYPE plain PLOT_DEGREE_FORMAT D TICK_LENGTH 0.3c LABEL_FONT_SIZE 12 ANOT_FONT_SIZE 10  HEADER_FONT 1 ANOT_FONT 1 LABEL_FONT 1 HEADER_FONT_SIZE 18 FRAME_PEN 2p TICK_PEN 2p MEASURE_UNIT inch
psbasemap -JM7.5i -R-122/-114.5/32/37 -B2/2WesN   -K  -V   >   socal_map.ps
grdimage /data2/Datalib/SC/w140n40.Bathmetry.srtm.swap.grd -JM -R -C/data2/Datalib/SC/w140n40.Bathmetry.srtm.swap.cpt -I/data2/Datalib/SC/w140n40.Bathmetry.srtm.swap.int  /data2/Datalib/SC/w140n40.Bathmetry.srtm.swap.grd  -I/data2/Datalib/SC/w140n40.Bathmetry.srtm.swap.int   -K -O -V   >>   socal_map.ps
pscoast -JM -R -W1.5 -Na/1.0p,255/255/255  -Dh   -K -O -V   >>   socal_map.ps
psxy /data2/Datalib/SC//jennings.xy -JM -R -W0.5  -M   -K -O -V   >>   socal_map.ps
psxy event_info/station.gmt -JM -R -St0.07  -W0.3  -G255/0/0   -K -O -V   >>   socal_map.ps
psmeca -JM -R -Sm0.20 -L2   -K -O -V <<EOF >> socal_map.ps
-120.0142 34.4135 9.6700 -3.253867e+20 -1.458079e+21 1.783466e+21 6.90522e+20 3.79331e+18 1.061939e+20 1 0 0
-116.8413 34.3533 9.8900 4.184e+20 -7.024e+20 2.838e+20 -1.002e+20 -3.875e+20 2.001e+20 1 0 0
-120.4963 35.9437 9.6560 -4.3e+19 -1.021e+21 1.063e+21 -3.43e+20 -5.21e+20 5.13e+20 1 0 0
-119.1958 35.0023 13.4000 4.93e+20 -8.62e+20 3.68e+20 -7.24e+20 -7.43e+20 1.195e+21 1 0 0
-120.4792 35.9269 9.2680 7.290196e+19 -3.397228e+21 3.324326e+21 -1.251663e+20 -4.940428e+20 5.555823e+20 1 0 0
-116.7725 34.0198 17.6000 1.049e+20 -3.574e+20 2.524e+20 4.541e+20 3.424e+20 2.559e+20 1 0 0
-116.7715 34.0182 18.5800 2.65e+20 -1.059e+21 7.93e+20 9.76e+20 7.04e+20 4.87e+20 1 0 0
-118.145 32.497 10.0000 1.321e+21 -1.405e+21 8.3e+19 -1e+19 -3.68e+20 -5.3e+19 1 0 0
-115.8518 32.705 10.3400 -5.6e+19 -2.296e+20 2.855e+20 1.254e+20 5.64e+19 -2.48e+19 1 0 0
-116.7903 33.9217 20.8400 1.052e+20 -3.29e+20 2.237e+20 1.214e+20 2.141e+20 2.51e+19 1 0 0
-116.0402 32.7333 17.9600 -7.04e+19 -6.014e+20 6.716e+20 -1.153e+20 5.83e+19 -5.68e+19 1 0 0
-116.052 32.7165 15.3300 1.19e+20 -1.835e+21 1.715e+21 -3.83e+20 -7.95e+20 -9.03e+20 1 0 0
-116.0448 33.7063 14.6500 -7.53e+19 -7.967e+20 8.719e+20 4.067e+20 4.038e+20 -2.65e+19 1 0 0
-116.2947 32.9945 3.6500 -5.24e+19 -2.38e+20 2.903e+20 -2.61e+19 2.87e+19 -2.19e+19 1 0 0
-116.1357 33.222 22.2000 4.947382e+19 -1.174323e+21 1.124849e+21 -3.616609e+19 2.409231e+20 6.489642e+19 1 0 0
-117.8735 36.0223 2.8600 7.286071e+19 -1.588659e+21 1.515799e+21 1.002301e+21 3.323824e+20 4.659379e+21 1 0 0
-117.4629 34.2696 10.5340 2.889e+20 -2.62e+20 -2.7e+19 2.808e+20 2.041e+20 3.14e+20 1 0 0
-117.8723 35.9915 3.8700 -2.283795e+20 -3.408665e+20 5.69246e+20 -6.121705e+20 2.165941e+20 1.584125e+21 1 0 0
-117.865 35.9783 2.6160 4.21e+19 -1.937e+20 1.514e+20 8.24e+19 5.2e+18 1.777e+20 1 0 0
-115.7451 32.5553 10.0010 -2.678e+20 -1.48e+19 2.824e+20 -7.17e+19 2.241e+20 1.93e+19 1 0 0
-119.3317 33.6678 14.5600 3.219e+21 -1.685e+21 -1.535e+21 -1.8e+19 -1.34e+20 2.043e+21 1 0 0
-118.0758 35.7057 3.2670 -1.42e+19 -2.033e+20 2.173e+20 2.74e+19 -5.98e+19 -6.58e+19 1 0 0
-117.4322 34.1653 7.9810 -1.24e+19 -2.882e+20 3.005e+20 7.47e+19 -4.96e+19 9.86e+19 1 0 0
-116.846 34.3103 3.6300 -4.8e+21 -1.941e+22 2.42e+22 -8.44e+21 -4.65e+21 -3.2e+20 1 0 0
-116.8547 34.3208 5.0100 -1.487e+20 -1.725e+20 3.21e+20 -1.819e+20 -1.515e+20 -2.573e+20 1 0 0
-116.8482 34.3097 4.0400 2.936e+21 -2.758e+21 -1.79e+20 -1.37e+20 -3.07e+20 -6.5e+19 1 0 0
-116.8407 34.3137 3.1300 -2.12e+20 -1.472e+21 1.682e+21 -4.02e+20 -2.32e+20 1.24e+20 1 0 0
-116.1303 34.3582 7.6800 -1.9e+20 -2.403e+21 2.591e+21 -3.39e+20 1.22e+20 6.6e+19 1 0 0
-115.5538 32.9475 9.8900 -3.7e+19 -2.004e+21 2.04e+21 -6.15e+20 2.94e+20 3.37e+20 1 0 0
-115.5472 32.9443 9.8400 -1.042662e+19 -4.162145e+20 4.266411e+20 6.801518e+19 -1.635967e+19 2.181727e+20 1 0 0
-118.2692 36.4782 12.4900 -2.068e+20 -9.1e+19 2.977e+20 -1.102e+20 9.32e+19 -1.04e+19 1 0 0
-117.5664 35.6352 2.3010 1.46e+19 -2.211e+20 2.064e+20 -9.1e+18 -8.72e+19 7.77e+19 1 0 0
-115.7441 32.5392 9.7670 -1.284871e+20 -2.15859e+20 3.44346e+20 2.669096e+20 1.814099e+20 7.406408e+19 1 0 0
-116.052 33.7152 12.4700 7.283311e+19 -5.08926e+20 4.360929e+20 2.438615e+20 2.939718e+20 -1.30456e+19 1 0 0
-119.4365 34.3885 8.1100 1.507043e+21 -1.543834e+21 3.67914e+19 1.328969e+21 3.071472e+20 6.11379e+18 1 0 0
-117.4478 34.1358 5.7070 2.98e+19 -2.646e+20 2.347e+20 6.21e+19 1.018e+20 3.69e+19 1 0 0
-120.5134 35.9528 10.4120 3.917519e+20 -2.260615e+22 2.22144e+22 -1.159868e+21 -2.996999e+21 4.563199e+21 1 0 0
-118.6292 35.3852 7.3000 -5.795192e+20 -1.857143e+22 1.915095e+22 8.38815e+21 -2.52875e+21 -1.911421e+22 1 0 0
-120.5403 35.9821 9.5900 -170172.7 -1.131021e+22 1.131021e+22 9.652735e+20 -9.995699e+20 3.949613e+20 1 0 0
-120.8108 35.5473 6.6500 6.701e+20 -9.761e+20 3.058e+20 7.115e+20 3.405e+20 7.491e+20 1 0 0
-117.442 34.1225 4.2450 3.33e+19 -2.66e+20 2.326e+20 3.43e+19 6.48e+19 1.01e+19 1 0 0
-117.4438 34.1272 4.7500 2.8e+20 -2.113e+21 1.831e+21 4.42e+20 4.66e+20 2.18e+20 1 0 0
-116.3912 33.9578 8.3200 -4.017387e+19 -1.234203e+21 1.274377e+21 3.159488e+20 -4.826834e+20 8.218002e+20 1 0 0
-116.2515 33.2884 4.4400 1.577272e+19 -2.948555e+20 2.790828e+20 3.236759e+19 -1.483192e+20 3.304975e+20 1 0 0
-116.8122 32.7233 19.8400 8.52e+19 -4.435e+20 3.581e+20 -6.56e+19 -7.97e+19 1.953e+20 1 0 0
-119.194 34.9987 10.5400 7.928e+21 -8.145e+21 2.16e+20 -5.204e+21 -4.273e+21 -5.69e+20 1 0 0
-116.5675 33.538 14.1300 1.13e+22 -4.618e+22 3.487e+22 1.053e+22 2.575e+22 -1.506e+22 1 0 0
-117.0072 34.0612 14.2500 1.242e+22 -1.797e+22 5.54e+21 6.61e+21 8.84e+21 4.84e+21 1 0 0
-117.0232 34.0615 13.8300 1.77e+19 -2.632e+20 2.454e+20 1.069e+20 1.11e+20 2.541e+20 1 0 0
-119.7527 33.6853 3.5100 1.923e+20 -3.631e+20 1.707e+20 -1.758e+20 -9.04e+19 5.73e+20 1 0 0
-118.0652 36.1488 3.0200 -5.5e+20 -6.46e+20 1.195e+21 -6.38e+20 1.86e+20 1e+20 1 0 0
-115.6207 33.1544 4.2150 143604.1 -6.917155e+21 6.917155e+21 1.035359e+21 5.505099e+20 4.66568e+21 1 0 0
-115.6098 33.1639 4.0470 98223.83 -4.73127e+21 4.73127e+21 7.081756e+20 3.765436e+20 3.191282e+21 1 0 0
-115.6157 33.1548 4.7850 9.61e+20 -3.728e+21 2.765e+21 -1.08e+20 6.07e+20 1.781e+21 1 0 0
-115.5924 33.1748 3.9460 23756.22 -1.144295e+21 1.144295e+21 1.712779e+20 9.107008e+19 7.718369e+20 1 0 0
-115.5969 33.1712 4.8450 8.88e+20 -2.971e+21 2.082e+21 -2.39e+20 3.55e+20 3.96e+21 1 0 0
-115.6168 33.1538 4.5550 18096.57 -8.716798e+20 8.716798e+20 1.304729e+20 6.937366e+19 5.879555e+20 1 0 0
-115.6064 33.1643 2.8170 134151.7 -6.461852e+21 6.461852e+21 9.672088e+20 5.14274e+20 4.358574e+21 1 0 0
-115.6295 33.1479 4.7930 437807.6 -2.108842e+22 2.108842e+22 3.156512e+21 1.678347e+21 1.422432e+22 1 0 0
-116.8393 32.5112 18.1800 3.04e+19 -1.963e+20 1.657e+20 -6.71e+19 -2.5e+19 1.376e+20 1 0 0
-116.026 33.1787 6.1430 3.81e+19 -1.047e+20 6.64e+19 -4.99e+19 7.6e+19 -1.085e+20 1 0 0
-119.0247 35.0178 10.1600 9.473e+21 -8.921e+21 -5.53e+20 -3.776e+21 1.201e+21 1.642e+21 1 0 0
-121.0838 35.65 5.4600 1.073e+21 -1.367e+21 2.93e+20 -1.19e+20 1e+19 6.67e+20 1 0 0
-117.545 35.1267 4.3750 -2.07e+19 -1.528e+20 1.733e+20 8.24e+19 6.12e+19 1.978e+20 1 0 0
-117.5828 35.6232 8.5900 -1.8e+18 -1.188e+20 1.205e+20 -1.095e+20 5.51e+19 3.54e+20 1 0 0
-116.022 33.245 4.9700 5.184251e+20 -2.075894e+21 1.557469e+21 -6.464972e+20 8.986633e+20 -1.090401e+21 1 0 0
-117.1103 33.8567 18.0600 -1.084e+20 -1.15e+20 2.232e+20 -1.638e+20 3.395e+20 6.26e+19 1 0 0
-116.0632 33.2663 9.6800 -6.2e+18 -4.114e+20 4.174e+20 -2.22e+19 8.35e+19 -3.445e+20 1 0 0
-115.9628 32.8423 3.8600 8.15e+19 -1.722e+20 9.06e+19 5.27e+19 8.61e+19 3.884e+20 1 0 0
-115.9672 32.8385 7.6300 4.67e+19 -4.443e+20 3.974e+20 6.55e+19 4.793e+20 7.988e+20 1 0 0
-115.9632 32.8318 5.8200 1.96e+19 -3.432e+20 3.235e+20 9.3e+19 1.01e+20 3.966e+20 1 0 0
-117.763 33.956 13.9000 8.342705e+22 -1.420774e+23 5.865038e+22 -1.909077e+22 7.404729e+22 -4.912553e+22 1 0 0
-116.2465 34.4368 7.7500 1.071e+22 5.181e+22 -6.254e+22 -4.679e+22 -1.08e+21 9.274e+22 1 0 0
-116.3978 34.8633 4.6700 -7.19e+20 -2.974e+21 3.691e+21 3.107e+21 3.19e+20 3.201e+21 1 0 0
-116.2888 34.8031 5.8330 13449.65 4.167661e+20 -4.167661e+20 -1.025302e+20 3.935767e+19 4.628656e+20 1 0 0
-116.3955 34.8735 3.3200 -3.38e+21 -1.529e+22 1.866e+22 7.4e+21 1.43e+21 1.428e+22 1 0 0
-116.371 34.8038 4.0900 -2.404e+20 2.525e+20 -1.23e+19 5.742e+20 -1.923e+20 6.768e+20 1 0 0
-117.6053 36.0882 3.4000 1.128969e+20 -2.3097e+21 2.196803e+21 6.051028e+20 -6.244466e+20 -7.048327e+20 1 0 0
-116.8417 34.3228 5.4200 -1.018e+20 -5.23e+20 6.247e+20 -1.63e+20 -6.49e+19 1.759e+20 1 0 0
-116.1448 34.3512 4.2800 -9.13271e+19 -2.32137e+21 2.412697e+21 -4.94389e+20 1.736222e+20 1.926568e+21 1 0 0
-116.3442 34.711 9.6400 7.903623e+19 -3.852611e+20 3.062249e+20 -7.280276e+19 7.507429e+20 1.065464e+21 1 0 0
-116.3952 34.8632 4.7200 2.044e+20 4.577e+20 -6.623e+20 3.56e+20 1.242e+20 6.589e+20 1 0 0
-116.2085 34.33 11.2100 3.297317e+19 -3.381612e+20 3.05188e+20 4.790077e+19 1.365544e+20 5.241836e+20 1 0 0
-116.406 34.862 2.9800 3.108024e+21 -3.108024e+21 4506310 1.46221e+22 3.867501e+21 3.679681e+22 1 0 0
-117.9045 36.4512 3.1960 -1.930595e+19 -1.117154e+21 1.13646e+21 4.240928e+20 3.552186e+20 1.281384e+20 1 0 0
-116.2707 34.52 2.7900 1.31e+20 1.151e+21 -1.283e+21 6.31e+20 -6.2e+19 9.09e+20 1 0 0
-116.298 34.7903 6.3700 -6.89e+19 4.546e+20 -3.859e+20 1.548e+20 -3.234e+20 8.904e+20 1 0 0
-116.4105 34.8343 7.2800 2.3e+19 -7.48e+20 7.24e+20 -3.68e+20 6.12e+20 2.42e+21 1 0 0
-117.0072 34.1048 4.2330 1.824675e+20 -1.16191e+21 9.794428e+20 2.124051e+20 -4.231121e+20 -4.438128e+20 1 0 0
-116.2635 34.5927 11.0400 1.589e+20 9.94e+19 -2.584e+20 2.105e+20 1.682e+20 6.397e+20 1 0 0
-117.2432 34.0588 16.5200 -9.7e+19 1.84e+20 -8.8e+19 -9.5e+19 -1.47e+20 1.746e+21 1 0 0
-117.601 36.085 0.1000 -1.506e+20 -6.116e+20 7.621e+20 8.945e+20 -1.421e+20 -6.96e+19 1 0 0
-115.3868 32.702 10.1800 9.422045e+18 -3.042395e+21 3.032973e+21 -1.321762e+20 2.413056e+20 5.404093e+20 1 0 0
-116.2704 34.8046 6.5740 2.715984e+19 6.521871e+19 -9.237856e+19 -5.032166e+19 -9.549752e+19 2.864628e+20 1 0 0
-118.0507 36.3267 4.9770 -4.385e+20 3.009e+20 1.375e+20 -4.599e+20 4.7e+19 3.855e+20 1 0 0
-115.5035 32.8898 9.4700 -1.662e+21 -4.58e+20 2.118e+21 8.1e+20 -3.16e+20 1.788e+21 1 0 0
-116.2985 34.7878 6.1600 -2.7e+20 2.286e+21 -2.017e+21 7.54e+20 -6.7e+20 4.003e+21 1 0 0
-116.7722 34.2673 5.6700 -3.378831e+19 -5.911568e+20 6.249451e+20 -1.520227e+20 4.035931e+19 -6.839581e+19 1 0 0
-119.0298 34.8942 14.8200 -81774.19 -1.148067e+21 1.148067e+21 -2.501387e+20 -6.191149e+20 -1.188858e+21 1 0 0
-118.4178 34.2833 7.0300 3.278101e+21 -1.3927e+21 -1.885401e+21 -3.327305e+20 9.145238e+20 1.654577e+21 1 0 0
-118.4173 34.2872 7.1900 5.067e+20 -6.357e+20 1.288e+20 2.258e+20 2.166e+20 4.605e+20 1 0 0
-116.9397 34.291 7.5800 -7e+19 -7.399e+21 7.468e+21 2.637e+21 -1.525e+21 -5.709e+21 1 0 0
-117.7091 33.8725 2.4550 3.36e+19 -1.785e+20 1.448e+20 -2.46e+19 8.5e+19 1.45e+19 1 0 0
-118.3256 35.9817 4.4480 -1.252e+21 -3.1e+19 1.282e+21 2.07e+20 4.62e+20 -1.4e+19 1 0 0
-118.0423 35.7957 9.0300 -3.52e+20 -8.81e+20 1.232e+21 6.24e+20 7.76e+20 -2.73e+20 1 0 0
-116.7523 34.0299 14.8350 1.629e+20 -2.801e+20 1.17e+20 1.72e+19 2.104e+20 -1.52e+20 1 0 0
-116.7605 34.2594 4.0680 -3.1e+18 -2.875e+20 2.905e+20 -2.413e+20 1.387e+20 -5.98e+19 1 0 0
-117.8682 36.0135 3.4200 -1317164 -2.072774e+22 2.072774e+22 1.055785e+22 2.052238e+21 5.130296e+22 1 0 0
-117.8753 36.014 2.3800 -6.795947e+21 -1.449699e+22 2.129294e+22 1.1594e+22 -3.594606e+21 1.720158e+22 1 0 0
-117.8632 36.0495 3.2900 -5.132e+20 -4.758e+20 9.888e+20 -5.707e+20 -4.99e+19 4.059e+20 1 0 0
-117.872 35.9896 3.6600 -1.655085e+20 -3.140082e+20 4.795166e+20 -3.790169e+20 6.835817e+19 6.868865e+20 1 0 0
-118.2513 32.792 4.9000 -4.57e+20 -3.695e+21 4.15e+21 2.3e+19 -7.08e+20 -2e+20 1 0 0
-118.318 32.734 10.0000 -4.52e+20 -3.569e+21 4.019e+21 3.38e+20 -1.06e+21 -3.57e+20 1 0 0
-118.3968 34.0527 7.7500 1.58e+20 -1.03e+21 8.71e+20 -8.53e+20 -4.57e+20 2.762e+21 1 0 0
-118.2795 33.9297 19.2400 3.201e+20 -1.207e+20 -1.995e+20 6.443e+20 -4e+19 2.842e+20 1 0 0
-116.5023 33.5112 15.5900 -1.05e+21 -1.577e+22 1.68e+22 1.515e+22 1.478e+22 2.02e+21 1 0 0
-115.71 33.307 7.2030 7.045322e+18 -5.844433e+20 5.77398e+20 -5.874298e+19 -8.611772e+19 3.522661e+18 1 0 0
-115.7012 33.317 7.8400 2.75e+20 -2.167e+21 1.891e+21 -2.75e+20 6.31e+20 3.58e+20 1 0 0
-116.7048 34.1178 8.2630 -9.99e+19 -3.586e+20 4.584e+20 2.664e+20 -1.427e+20 -1.001e+20 1 0 0
-117.7483 33.9552 12.4200 -8.3e+18 -3.972e+20 4.054e+20 -2.442e+20 -1.52e+20 -4.7e+18 1 0 0
-116.4308 33.3852 12.5600 3.7e+19 -1.01e+21 9.72e+20 2.78e+20 2.11e+20 6.1e+19 1 0 0
-118.6642 34.3638 11.3800 2.632981e+21 -2.580758e+21 -5.222245e+19 2.373766e+21 -3.960404e+19 4.6364e+20 1 0 0
-118.6642 34.367 11.2160 3.724e+20 -3.168e+20 -5.58e+19 4.228e+20 1.948e+20 2.802e+20 1 0 0
-118.6645 34.3655 11.5360 7.592071e+20 -8.322336e+20 7.30265e+19 3.134622e+20 2.502231e+20 8.017266e+18 1 0 0
-118.6669 34.3631 10.5540 6.84e+19 -1.33e+20 6.44e+19 3.562e+20 2.745e+20 -2.9e+18 1 0 0
-118.667 34.3647 10.7470 4.162103e+20 -4.18042e+20 1.831651e+18 3.277841e+20 4.119029e+19 -6.520666e+18 1 0 0
-116.7117 33.2065 9.2570 2.58e+19 -2.956e+20 2.697e+20 2.61e+19 7.88e+19 2.348e+20 1 0 0
-116.2952 34.5177 4.7700 -1.113e+21 4.7e+20 6.42e+20 -5.58e+20 -8.2e+19 -8.77e+20 1 0 0
-117.784 33.9133 9.4400 1.23e+20 -3.436e+21 3.312e+21 -3.96e+20 1.2e+19 -1.886e+21 1 0 0
-116.1118 33.2352 12.4700 -3.43e+19 -8.691e+20 9.032e+20 -2.536e+20 -2.84e+19 3.502e+20 1 0 0
-117.2913 35.9468 5.5900 5.353e+20 -7.373e+20 2.018e+20 4.09e+19 -5.844e+20 6.029e+20 1 0 0
-117.2917 35.949 5.5500 -7.78e+19 -2.976e+20 3.752e+20 -7.6e+19 -1.043e+20 3.15e+20 1 0 0
-116.5685 33.5125 13.4450 2.230852e+19 -1.869052e+20 1.645967e+20 -3.633966e+19 -6.511458e+19 1.85227e+20 1 0 0
-116.265 34.8068 7.4200 -8.28e+20 -3.56e+20 1.183e+21 -3.16e+20 5.22e+20 5.97e+21 1 0 0
-118.6632 35.3152 4.2170 -8.32e+20 -4.9e+19 8.8e+20 1.019e+21 -6.96e+20 -7.15e+20 1 0 0
-118.6585 35.3128 3.9700 -2.503e+21 -1.227e+21 3.729e+21 2.018e+21 -6.04e+20 -1.198e+21 1 0 0
-116.6665 34.6172 8.4000 2.016e+20 -1.347e+20 -6.7e+19 7.39e+19 -2.456e+20 -4.536e+20 1 0 0
-118.6509 34.4034 13.5690 9.4e+18 -4.8e+19 3.85e+19 6.094e+20 1.266e+20 2.117e+20 1 0 0
-115.2837 32.562 7.4990 6.862199e+19 -1.193572e+21 1.12495e+21 -2.949452e+20 -3.974688e+20 6.150361e+18 1 0 0
-121.0428 35.6493 7.7500 7.5475e+21 -7.538307e+21 -9.192665e+18 1.603293e+21 -5.598824e+19 2.632435e+20 1 0 0
-120.8385 35.5487 7.5400 2.585e+21 -3.143e+21 5.57e+20 3.03e+20 1.808e+21 1.242e+21 1 0 0
-119.1412 35.0118 13.5200 8.963e+21 -8.483e+21 -4.82e+20 4.61e+20 2.058e+21 -1.914e+21 1 0 0
EOF
pstext /data2/Datalib/SC/socal_topo_labs.xyz -JM -R -B  -G250  -S1p,0   -K -O -V   >>   socal_map.ps
pstext -JM -R -B  -G0/0/255  -D0/0.12   -K -O -V   <<EOF  >>   socal_map.ps
-120.0142 34.4135 7 0 4 CM 10006857
-116.8413 34.3533 7 0 4 CM 10059745
-120.4963 35.9437 7 0 4 CM 10063349
-119.1958 35.0023 7 0 4 CM 10097009
-120.4792 35.9269 7 0 4 CM 10100053
-116.7725 34.0198 7 0 4 CM 10148369
-116.7715 34.0182 7 0 4 CM 10148421
-118.1450 32.4970 7 0 4 CM 10148829
-115.8518 32.7050 7 0 4 CM 10186185
-116.7903 33.9217 7 0 4 CM 10187953
-116.0402 32.7333 7 0 4 CM 10207681
-116.0520 32.7165 7 0 4 CM 10215753
-116.0448 33.7063 7 0 4 CM 10223765
-116.2947 32.9945 7 0 4 CM 10226877
-116.1357 33.2220 7 0 4 CM 10230869
-117.8735 36.0223 7 0 4 CM 10964587
-117.4629 34.2696 7 0 4 CM 10972299
-117.8723 35.9915 7 0 4 CM 10992159
-117.8650 35.9783 7 0 4 CM 11671240
-115.7451 32.5553 7 0 4 CM 12456160
-119.3317 33.6678 7 0 4 CM 12659440
-118.0758 35.7057 7 0 4 CM 12887732
-117.4322 34.1653 7 0 4 CM 13692644
-116.8460 34.3103 7 0 4 CM 13935988
-116.8547 34.3208 7 0 4 CM 13936432
-116.8482 34.3097 7 0 4 CM 13936812
-116.8407 34.3137 7 0 4 CM 13938812
-116.1303 34.3582 7 0 4 CM 13945908
-115.5538 32.9475 7 0 4 CM 13966396
-115.5472 32.9443 7 0 4 CM 13970876
-118.2692 36.4782 7 0 4 CM 13986104
-117.5664 35.6352 7 0 4 CM 14007388
-115.7441 32.5392 7 0 4 CM 14072464
-116.0520 33.7152 7 0 4 CM 14073800
-119.4365 34.3885 7 0 4 CM 14077668
-117.4478 34.1358 7 0 4 CM 14079184
-120.5134 35.9528 7 0 4 CM 14095540
-118.6292 35.3852 7 0 4 CM 14095628
-120.5403 35.9821 7 0 4 CM 14096196
-120.8108 35.5473 7 0 4 CM 14096736
-117.4420 34.1225 7 0 4 CM 14116920
-117.4438 34.1272 7 0 4 CM 14116972
-116.3912 33.9578 7 0 4 CM 14118096
-116.2515 33.2884 7 0 4 CM 14133048
-116.8122 32.7233 7 0 4 CM 14137160
-119.1940 34.9987 7 0 4 CM 14138080
-116.5675 33.5380 7 0 4 CM 14151344
-117.0072 34.0612 7 0 4 CM 14155260
-117.0232 34.0615 7 0 4 CM 14158696
-119.7527 33.6853 7 0 4 CM 14165408
-118.0652 36.1488 7 0 4 CM 14169456
-115.6207 33.1544 7 0 4 CM 14178184
-115.6098 33.1639 7 0 4 CM 14178188
-115.6157 33.1548 7 0 4 CM 14178212
-115.5924 33.1748 7 0 4 CM 14178236
-115.5969 33.1712 7 0 4 CM 14178248
-115.6168 33.1538 7 0 4 CM 14179288
-115.6064 33.1643 7 0 4 CM 14179292
-115.6295 33.1479 7 0 4 CM 14179736
-116.8393 32.5112 7 0 4 CM 14181056
-116.0260 33.1787 7 0 4 CM 14183744
-119.0247 35.0178 7 0 4 CM 14186612
-121.0838 35.6500 7 0 4 CM 14189556
-117.5450 35.1267 7 0 4 CM 14204000
-117.5828 35.6232 7 0 4 CM 14219360
-116.0220 33.2450 7 0 4 CM 14236768
-117.1103 33.8567 7 0 4 CM 14239184
-116.0632 33.2663 7 0 4 CM 14255632
-115.9628 32.8423 7 0 4 CM 14263544
-115.9672 32.8385 7 0 4 CM 14263712
-115.9632 32.8318 7 0 4 CM 14263768
-117.7630 33.9560 7 0 4 CM 14383980
-116.2465 34.4368 7 0 4 CM 3320736
-116.3978 34.8633 7 0 4 CM 3320884
-116.2888 34.8031 7 0 4 CM 3321426
-116.3955 34.8735 7 0 4 CM 3321590
-116.3710 34.8038 7 0 4 CM 7177729
-117.6053 36.0882 7 0 4 CM 7179710
-116.8417 34.3228 7 0 4 CM 9105672
-116.1448 34.3512 7 0 4 CM 9111353
-116.3442 34.7110 7 0 4 CM 9112735
-116.3952 34.8632 7 0 4 CM 9113909
-116.2085 34.3300 7 0 4 CM 9114763
-116.4060 34.8620 7 0 4 CM 9114812
-117.9045 36.4512 7 0 4 CM 9116921
-116.2707 34.5200 7 0 4 CM 9117942
-116.2980 34.7903 7 0 4 CM 9120741
-116.4105 34.8343 7 0 4 CM 9122706
-117.0072 34.1048 7 0 4 CM 9128775
-116.2635 34.5927 7 0 4 CM 9130422
-117.2432 34.0588 7 0 4 CM 9140050
-117.6010 36.0850 7 0 4 CM 9141142
-115.3868 32.7020 7 0 4 CM 9146641
-116.2704 34.8046 7 0 4 CM 9147453
-118.0507 36.3267 7 0 4 CM 9151609
-115.5035 32.8898 7 0 4 CM 9154092
-116.2985 34.7878 7 0 4 CM 9155518
-116.7722 34.2673 7 0 4 CM 9169867
-119.0298 34.8942 7 0 4 CM 9171679
-118.4178 34.2833 7 0 4 CM 9173365
-118.4173 34.2872 7 0 4 CM 9173374
-116.9397 34.2910 7 0 4 CM 9627721
-117.7091 33.8725 7 0 4 CM 9644101
-118.3256 35.9817 7 0 4 CM 9644345
-118.0423 35.7957 7 0 4 CM 9653493
-116.7523 34.0299 7 0 4 CM 9655209
-116.7605 34.2594 7 0 4 CM 9666905
-117.8682 36.0135 7 0 4 CM 9674049
-117.8753 36.0140 7 0 4 CM 9674213
-117.8632 36.0495 7 0 4 CM 9686565
-117.8720 35.9896 7 0 4 CM 9688709
-118.2513 32.7920 7 0 4 CM 9695397
-118.3180 32.7340 7 0 4 CM 9695549
-118.3968 34.0527 7 0 4 CM 9703873
-118.2795 33.9297 7 0 4 CM 9716853
-116.5023 33.5112 7 0 4 CM 9718013
-115.7100 33.3070 7 0 4 CM 9722529
-115.7012 33.3170 7 0 4 CM 9722633
-116.7048 34.1178 7 0 4 CM 9734033
-117.7483 33.9552 7 0 4 CM 9735129
-116.4308 33.3852 7 0 4 CM 9742277
-118.6642 34.3638 7 0 4 CM 9753485
-118.6642 34.3670 7 0 4 CM 9753489
-118.6645 34.3655 7 0 4 CM 9753497
-118.6669 34.3631 7 0 4 CM 9753949
-118.6670 34.3647 7 0 4 CM 9755013
-116.7117 33.2065 7 0 4 CM 9774569
-116.2952 34.5177 7 0 4 CM 9775765
-117.7840 33.9133 7 0 4 CM 9818433
-116.1118 33.2352 7 0 4 CM 9826789
-117.2913 35.9468 7 0 4 CM 9828889
-117.2917 35.9490 7 0 4 CM 9829213
-116.5685 33.5125 7 0 4 CM 9853417
-116.2650 34.8068 7 0 4 CM 9854597
-118.6632 35.3152 7 0 4 CM 9882325
-118.6585 35.3128 7 0 4 CM 9882329
-116.6665 34.6172 7 0 4 CM 9930549
-118.6509 34.4034 7 0 4 CM 9941081
-115.2837 32.5620 7 0 4 CM 9944301
-121.0428 35.6493 7 0 4 CM 9967901
-120.8385 35.5487 7 0 4 CM 9968977
-119.1412 35.0118 7 0 4 CM 9983429
EOF
psxy -JX2 -R0/1/0/1   -O  -V   <<EOF  >>   socal_map.ps
EOF
