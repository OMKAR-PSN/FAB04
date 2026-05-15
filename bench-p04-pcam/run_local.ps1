# Runner to launch the local P-04 bench using the adapter in fab04\adapters
Push-Location C:\Users\Omkar\Desktop\FAB04\fab04
$env:PYTHONPATH = 'C:\Users\Omkar\Desktop\FAB04\fab04'
C:\python313\python.exe .\out\bench-p04-pcam\self_check.py --adapter adapters.myteam:Engine --quick
Pop-Location
