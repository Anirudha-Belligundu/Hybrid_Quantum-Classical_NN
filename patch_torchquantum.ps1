# Fix utils.py
$utils_path = "$env:LOCALAPPDATA\Programs\Python\Python310\Lib\site-packages\torchquantum\utils.py"
(Get-Content $utils_path) -replace 'from qiskit import IBMQ', 'from qiskit.providers.ibmq import IBMQ' | Set-Content $utils_path
(Get-Content $utils_path) -replace 'from qiskit\.providers\.aer\.noise\.device\.parameters import gate_error_values', '# Removed problematic import' | Set-Content $utils_path

# Fix functional.py
$functional_path = "$env:LOCALAPPDATA\Programs\Python\Python310\Lib\site-packages\torchquantum\functional.py"
(Get-Content $functional_path) -replace 'gate_error_values', 'None' | Set-Content $functional_path

Write-Host "TorchQuantum patched successfully!" -ForegroundColor Green