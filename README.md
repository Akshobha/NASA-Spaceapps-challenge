# NASA-Spaceapps-challenge
Hello everyone, this is a repo containing my model pt file.
I wrote a ML pipeline which leverages this github repo for training a model for exoplanet classification.
Here are some guidelines you need to follow while using my code:
1. Make sure that git for desktop is installed in your device.
2. Your device runs on Windows as my code saves its results in a devices C drive.
3. Your device should have python installed and it should have Gitpython, Scikit-learn, Pytorch, Pandas and Numpy installed.
4. You should delete your local repo of this everytime you use this prototype as it git clones this repo everytime it's used.
5. Make sure your device is plugged in for optimal performance as this prototype runs an FNN model locally on your device.
6. Follow the instructions given in my apps UI.
7. Your input files should have the following variables(with the same names as what's in this file):
8. pl_trandurh:Planet Transit Duration Value [hours]
9. pl_rade:Planet Radius Value [R_Earth]
10. st_dist: Stellar Distance [pc]
11. st_tmag: TESS Magnitude
12. pl_orbper: Planet Orbital Period Value [days]
13. st_pmraerr1:PMRA Upper Unc [mas/yr]
14. st_pmraerr2: PMRA Lower Unc [mas/yr]
15. st_pmdec: PMDec [mas/yr]
16. st_pmdecerr1:PMDec Upper Unc [mas/yr]
17. st_pmdecerr2:PMDec Limit Flag
18. st_pmdeclim:PMDec Limit Flag
19. pl_tranmid:Planet Transit Midpoint Value [BJD]
20. pl_tranmiderr1: Planet Transit Midpoint Upper Unc [BJD]
21. pl_tranmiderr2: Planit Transit Midpoint Lower Unc [BJD]
22. pl_tranmidlim: Planet Transit Midpoint Limit Flag
23. pl_orbpererr1: Planet Orbital Period Upper Unc [days]
24. pl_orbpererr2: Planet Orbital Period Lower Unc [days]
25. pl_trandurherr1: Planet Transit Duration Upper Unc [hours]
26. pl_trandurherr2:Planet Transit Duration Lower Unc [hours]
27. pl_radeerr1: Planet Radius Upper Unc [R_Earth]
28. pl_radeerr2: Planet Radius Lower Unc [R_Earth]
29. st_disterr1: Stellar Distance Upper Unc [pc]
30. st_disterr2:Stellar Distance Lower Unc [pc]
31. st_teff: Stellar Effective Temperature Value [K]
32. st_tefferr1:Stellar Effective Termperature Upper Unc [K]
33. pl_trandurh: Planet Transit Duration Value [hours]

Thank you for using my code and I hope you enjoyed using as much as I enjoyed developing it!
